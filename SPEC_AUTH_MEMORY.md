# NeuroAppAgentic — user login and per-user memory

Adds two things to the agentic app: an identity, and a memory scoped to it. Nothing in the
graph's control flow changes. `SPEC.md` remains the spec for retrieval and self-correction;
this document only covers what sits around it.

## 1. Why identity has to come first

mem0 is keyed by `user_id`. Without authentication that key can only come from the client —
a text box, a URL parameter, a cookie the browser owns — and any of those let one visitor
read another's memories by typing their name. A per-user memory store with unauthenticated
user IDs is not a weaker version of this feature, it is a data leak. So login is not a
convenience wrapper here; it is the thing that makes the memory safe to have.

## 2. Login — `st.login()`, native OIDC

Streamlit 1.42+ ships OIDC login. The installed version is **1.62.0**, so it is available
with no new dependency beyond `Authlib`.

```
.streamlit/secrets.toml

[auth]
redirect_uri = "https://<app>.streamlit.app/oauth2callback"
cookie_secret = "<64 random hex chars>"

[auth.google]
client_id     = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

The app gates on `st.user.is_logged_in` and reads `st.user.sub` / `st.user.email`.

**Why this and not a user table.** A `users` table means storing password hashes, writing a
reset flow, and owning a breach surface — in a clinical-facing app, for zero product value.
Google carries the credential; this app never sees one.

| choice | cost | why not |
| --- | --- | --- |
| **`st.login()` + Google OIDC** | free | — chosen |
| `streamlit-authenticator` | free | app owns password hashes; no free persistent store on Community Cloud to put them in |
| Auth0 / Clerk free tier | free tier | a second vendor and a second dashboard for one OIDC redirect Google already serves |
| Supabase Auth | free tier | reasonable, but only worth it if Supabase is already the datastore — see §6 |

**Free-tier catches, all real:**

- Community Cloud has **no `.env`**; these go in the app's Secrets panel. The app already
  copies `st.secrets` into `os.environ` at startup, but `[auth]` is read by Streamlit itself
  and must stay a real TOML table — do not flatten it.
- `redirect_uri` must match the deployed URL **exactly** and must be registered in the
  Google Cloud console. Local and deployed need two entries, or two OAuth clients.
- The Google consent screen in **Testing** mode allows ~100 named test users and is free
  forever. `openid`/`email`/`profile` are non-sensitive scopes, so publishing needs no
  verification review — but publishing also means anyone with a Google account can log in.
  For a clinician-facing tool, stay in Testing and list users explicitly.
- `cookie_secret` rotation logs everyone out. Set it once.

## 3. Identity that survives sessions — the user key is `sub`, not email

The whole point of the memory is that the person who logs in tomorrow, from a different
browser, after the app has been redeployed twice, is recognised as the same person. That is
a property of the key, not of the store, so the key is chosen for exactly this:

```python
user_id = hashlib.sha256((PEPPER + st.user.sub).encode()).hexdigest()[:32]
```

**Why `sub` and nothing else.** Google's `sub` is a unique identifier for the Google account
that is never reused and does not change — not when the user renames their account, changes
their email, signs in from a phone, or clears every cookie they own. It is the only value in
the token with that guarantee. Email is mutable: a clinician who changes their address would
silently orphan every preference written under the old one and start over as a stranger.
`st.user.email` is therefore display-only and never leaves the process.

**What continuity depends on, in order:**

| layer | persists? | what breaks it |
| --- | --- | --- |
| `st.user.sub` | forever, per Google account | switching identity provider (Google → Microsoft) mints a different subject |
| `PEPPER` → `user_id` | as long as it is never rotated | rotating it renames every user at once — mass amnesia, silent |
| mem0 vector store | only if hosted (§5) | Community Cloud's disk is wiped on reboot and redeploy |
| Streamlit auth cookie | ~30 days | expiry logs the user out; it does **not** change who they are — re-login restores the same `user_id` |

Three consequences worth stating plainly, because each is a way to lose every user's memory
without an error appearing anywhere:

- **`PEPPER` is a permanent secret, not a rotatable one.** It lives in `st.secrets`, is set
  once, and is documented as un-rotatable. If it must ever change, it is a migration, not a
  config edit. (Dropping the pepper entirely is defensible — plain `sha256(sub)` is not
  reversible to anything useful — and removes the footgun. The pepper only buys protection
  against someone who already holds the store *and* a candidate list of Google subject IDs.)
- **The store must be hosted.** §6 treats this as a preference between backends; it is not.
  A local Chroma on Community Cloud satisfies every line of this spec except the one that
  matters, and fails it invisibly — it works all session and forgets on the next deploy.
- **A separate `sha256(sub)` fingerprint is written into memory metadata.** The raw OIDC
  subject never leaves the app, while a provider or pepper migration remains a backfill
  rather than a total loss.

Logging in is what re-establishes identity. There is no anonymous mode with a fallback key:
a session-generated ID would look like it worked, accumulate preferences for one afternoon,
and lose them — which is worse than having no memory at all.

## 4. What is allowed into memory — the hard boundary

This is the decision that matters most, because the failure mode is regulatory, not
technical.

**Only preferences are stored. Clinical content never is.**

A neurology assistant's conversation contains, in practice, patient-shaped text: a
clinician types a case to ask about it. Handing that verbatim to a hosted memory service is
a disclosure the app has no basis for. So writes go through a fixed schema, and anything
that does not fit is dropped:

| stored | example |
| --- | --- |
| answer style | "prefers bulleted answers", "wants dosing in mg/kg" |
| specialty focus | "works in neuro ICU", "interested in ALS" |
| citation depth | "always wants page numbers" |
| source preference | "prefers Bradley's over Harrison's" |

| never stored |
| --- |
| the question text verbatim |
| the generated answer |
| retrieved chunks |
| any name, age, date, MRN or free-text case description |

Enforcement is two-layered: a **schema-constrained extraction prompt** (the LLM may only
emit typed preference facts) and then the existing `guard()` redaction applied to each
candidate string **before** it is sent to mem0. Redaction after the fact is not a control;
it must run on the write path.

## 5. Configuring extraction — categories, instructions, exclusions

§4 states the boundary. This section is how it is expressed and enforced, and it is the only
part of the memory the app is expected to tune over time.

### 5.1 The taxonomy is code, not a dashboard setting

`src/memory_schema.py` is the single source of truth and holds three things: `CATEGORIES`,
`CUSTOM_INSTRUCTIONS`, and `SCHEMA_VERSION`. Where the backend also accepts a hosted copy
(§6), the app pushes this file at startup — the dashboard is a mirror, never the authority.

Two reasons, both consequences of how mem0 actually behaves:

- **Categorisation applies at ingestion only.** Changing the list does not re-tag memories
  already written. A taxonomy edit is therefore a migration, which is a thing that belongs
  in version control with a version number attached — hence `SCHEMA_VERSION`, stamped into
  every memory's metadata alongside the subject fingerprint from §3. Re-tagging is then a backfill query,
  not an archaeology exercise.
- **Extraction rules set in a web console are invisible state that changes model output.**
  Same argument as §10: a run whose behaviour depends on what someone last typed into a
  dashboard is not reproducible, and the drift shows up as "the assistant got worse" with
  nothing in the diff.

### 5.2 Categories

Six, closed set. Anything that does not fit one is not stored — the taxonomy *is* the filter.

| category | captures | example fact |
| --- | --- | --- |
| `answer_style` | shape and length of the answer | "prefers bulleted answers over prose" |
| `citation_depth` | how much sourcing to show | "always wants page numbers" |
| `source_preference` | which texts to lean on | "prefers Bradley's over Harrison's" |
| `unit_convention` | how to express quantities | "wants dosing in mg/kg" |
| `terminology_level` | jargon tolerance | "expand abbreviations on first use" |
| `clinical_focus` | subspecialty or setting, as a standing interest | "works in neuro ICU" |

`clinical_focus` is the one category that touches clinical ground, and it is deliberately
narrow: a *standing* interest of the clinician, never anything about a case in front of
them. "Works in neuro ICU" is a preference; "asking about a 54-year-old with status
epilepticus" is a patient and is dropped.

Categories are not decoration. They are what makes the memory bounded at read time: the
generation prompt pulls **at most two facts per category**, so a user with sixty memories
still contributes a fixed-size style block instead of steadily eating the context window
that retrieval needs.

### 5.3 Instructions

`CUSTOM_INSTRUCTIONS` is a single string with three parts — scope, exclusions, and few-shot
examples — and the examples carry most of the weight:

```
Extract ONLY standing preferences of the clinician using this assistant.
Assign each fact exactly one category from: answer_style, citation_depth,
source_preference, unit_convention, terminology_level, clinical_focus.

NEVER extract: patient details of any kind (age, sex, name, MRN, dates, presenting
complaint, case narrative), the clinical question itself, the assistant's answer, or
retrieved source text. A message describing a patient contains no extractable facts.

Input:  "Can you keep these shorter? Bullets are fine, and I always want the page number."
Output: {"facts": ["prefers short bulleted answers", "always wants page numbers"]}

Input:  "54F with status epilepticus refractory to lorazepam, what's next?"
Output: {"facts": []}

Input:  "I'm in the neuro ICU so assume ventilated patients."
Output: {"facts": ["works in neuro ICU"]}
```

The second example is the important one. Most of the traffic through this app looks like
that, and an extractor without an explicit negative case will happily decide a patient
presentation is a fact worth remembering.

### 5.4 Instructions are steering, not enforcement

This is the part to be honest about: `CUSTOM_INSTRUCTIONS` is a prompt. It is the same class
of control as the `grade` and `verify` gates, and `SPEC.md` already establishes what those
are worth — they fail, and the design assumes they fail. A prompt cannot be the thing
standing between a patient description and a third-party datastore.

So every candidate fact passes a deterministic validator before `add()` is called, and the
validator, not the prompt, is what §4 relies on:

| check | on failure |
| --- | --- |
| category ∈ `CATEGORIES` | drop the fact |
| length ≤ 120 chars | drop the fact |
| deny-regex: digits+age patterns, dates, MRN-shaped tokens, person names | drop the fact |
| `guard()` redaction leaves the string unchanged | drop the fact |

Note the last row's direction. Redaction is used as a **detector**, not a cleaner: if
`guard()` wanted to change the string, that string had something in it that does not belong
in a preference, and the fix is to discard it rather than to store a redacted version. A
sanitised patient description is still a patient description.

Dropped facts are counted and logged by category — never logged verbatim. A rising drop rate
in `clinical_focus` is the signal that `CUSTOM_INSTRUCTIONS` needs another negative example,
and it is the only visibility this path gets, since the contents cannot be inspected.

### 5.5 Three layers, in order

1. **Deterministic pre-filter** — does this turn plausibly carry a preference at all? Gates
   whether extraction runs, so an ordinary clinical question costs zero extra model calls
   (§8).
2. **`CUSTOM_INSTRUCTIONS`** — steers what the extractor emits and how it is categorised.
   Probabilistic. Tunable. Not trusted.
3. **Validator** — deterministic, enforces §4, drops anything that does not conform.

Only layer 3 is a control. Layers 1 and 2 exist to make layer 3 rarely have to fire.

## 6. mem0 configuration — and where §5 lands on each

The two mem0 distributions do not expose §5's controls the same way, so the choice is now
partly forced by it:

| control | Mem0 Platform | mem0 OSS |
| --- | --- | --- |
| custom instructions | `client.project.update(custom_instructions=...)`, project-level, with a per-`add()` override | `custom_instructions` in the `Memory.from_config({...})` dict |
| custom categories | `custom_categories`, project-level or per-`add()`; mem0 tags each memory | **no first-class equivalent** |
| filtering by category | native, on the stored tag | on `metadata`, written by this app |

**Recommendation: mem0 OSS**, and carry categories in `metadata` rather than adopting the
Platform for them.

The reasoning is §4. The boundary is only worth writing down if the data does not leave
infrastructure this app controls, and §5.4 concludes that a prompt cannot be trusted to hold
that line — which means the validator has to run in this process, before the write, whatever
the backend is. Once that is true, Platform's categories buy an automatic tag the app is
already computing for itself in order to validate it.

Two further points push the same way. Platform categorisation is **asynchronous**, so a
newly written memory can read back with `categories: null` — usable for analytics, not for a
read-time cap that has to be correct on the next turn (§5.2). And per-call category lists
**replace** the project list rather than merging with it, which is a quiet way to lose the
taxonomy on one code path.

So: `CATEGORIES` from `src/memory_schema.py` is enforced by the validator, and written as
`metadata={"category": ..., "schema_version": ..., "subject_hash": ...}`. `search()` filters on it.
The taxonomy is a closed set of six, so metadata equality is the whole implementation.

**Chosen stack — `mem0ai` OSS:**

| component | choice | tier |
| --- | --- | --- |
| extraction LLM | `gemini-2.5-flash-lite` | Google AI Studio free tier — structured extraction runs in the app before validation |
| embedder | `models/gemini-embedding-001`, 768-d | Google AI Studio free tier; avoids a second local model on Community Cloud |
| vector store | Supabase pgvector | free tier, 500 MB |
| history store | mem0 local SQLite | ephemeral audit only; pgvector remains the authoritative preference store |

*Catches:* a free Supabase project **pauses after ~1 week of inactivity** and must be
resumed from the dashboard — a demo that sat idle will throw connection errors on first
load, so the memory read must fail open (§8). Free-tier Gemini has per-minute request limits
that the extraction call shares with generation; extraction is the one to shed under 429,
never the answer. The app cannot use mem0's internal OSS extraction for writes because it
stores the result before application validation can run. Instead, `CUSTOM_INSTRUCTIONS`
drives a local structured extraction call, the validator enforces §5.4, and accepted facts
are written with `infer=False`. This preserves validation-before-storage.

**If the Platform is chosen anyway** — for the managed store, which is a fair reason —
nothing in §5 changes except its plumbing: `memory_schema.py` is pushed via
`client.project.update(custom_categories=CATEGORIES, custom_instructions=CUSTOM_INSTRUCTIONS)`
at startup so the file stays authoritative (§5.1), and the validator still runs locally
before `add()`. Confirm the current free-tier memory and request caps on their pricing page
first, and accept that preference text then lives on their infrastructure.

Local development can point mem0's vector store at the on-disk Chroma already in `data/` and
skip Supabase entirely. **Community Cloud's filesystem is ephemeral** — it is wiped on every
reboot and redeploy — so a local store there is not persistence, it is a cache that silently
forgets. Per §3 this disqualifies it for the deployed app: cross-session recognition is the
requirement, and a store that empties on redeploy cannot meet it while still appearing to
work for the length of any single demo.

## 7. Where memory attaches to the graph

```
login ─→ load_prefs ─→ route → retrieve → grade → generate → verify → answer
                                                     ↑                   │
                                              prefs injected here        │
                                                                  write_memory (async)
```

Two rules, both inherited from the existing threat model in `SPEC.md` §4:

- **Preferences are fenced with `fence_untrusted()`,** like retrieved text. They are
  user-authored strings that an LLM rewrote — untrusted by both routes.
- **Preferences reach the generation prompt and nothing else.** Not the router, not `grade`,
  not `verify`, not the refusal floor. `SPEC.md` already argues that a poisoned chunk must
  not flip a control-flow decision; a stored preference is a *persistent* poisoned chunk,
  re-injected on every future turn. "Prefer answers even when evidence is thin" must not be
  a thing a user can write into their own refusal floor.

Concretely: preferences render as a style block in the generation system prompt — grouped by
category, at most two facts each (§5.2), so the block has a hard ceiling of twelve short
lines regardless of how long the user has been using the app. If mem0 is down, unreachable,
or slow, that block is empty and the answer is the current answer.

## 8. Failure and latency budget

- **Read fails open.** `search()` is wrapped; on any exception or timeout the graph proceeds
  with no preferences. A memory service outage must never take the assistant down.
- **Write is off the response path.** `add()` costs 1–2 extra LLM calls; it runs after the
  answer is rendered, never before. The user waits for retrieval and generation only.
- **Write is conditional.** Extraction runs only when the turn plausibly carries a
  preference — a cheap deterministic pre-filter — so a normal clinical question costs zero
  extra calls. `SPEC.md` §7 already flags 4–9 calls per question as the cost problem;
  memory must not make it 6–11.
- **One read per session, not per turn.** Preferences are fetched at login into
  `st.session_state` and reused.

## 9. Isolation on shared infrastructure

Community Cloud runs every visitor in **one process**. `@st.cache_resource` is therefore
global across users — the app already uses it for the backend, BM25 index and LLMs, which
is correct because those are user-independent.

**No user-scoped value may ever be cached with `@st.cache_resource` or `@st.cache_data`.**
Preferences, `user_id`, and `st.user` live in `st.session_state` only. A cached
`get_prefs()` would serve one clinician's memory to the next visitor, and it would look
like it worked.

## 10. Evaluation

**Memory is disabled in `eval/`.** Same rule, same reason as `guard()`: `SPEC.md` §4 keeps
guardrails off the evaluation path because they rewrite text and skew the judges. Per-user
preferences rewrite the generation prompt, so a run's score would depend on hidden state
belonging to whoever last used the app — the numbers would stop being reproducible and stop
being comparable to every figure already recorded.

`eval/run_eval.py` calls the graph with `prefs=None`. There is no environment flag to get
this wrong.

If preference-conditioned generation is ever to be measured, it needs its own harness with
fixed synthetic preference sets — not the live store.

## 11. Out of scope

- Roles, permissions, admin views. Every logged-in user is the same kind of user.
- Cross-user or team memory.
- Conversation history persistence — this spec stores preferences, not transcripts (§4).
- A user-facing editor for stored preferences. Read-only display of what is remembered, by
  category, plus a delete-everything button, is in scope; editing individual facts is not.
- Learned or LLM-proposed categories. The taxonomy is a closed set of six, changed only by
  editing `memory_schema.py` and bumping `SCHEMA_VERSION` (§5.1).
- Any change to routing, retrieval, the gates, or the refusal floor.
