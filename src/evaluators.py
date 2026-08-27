"""LLM-judged and deterministic evaluators, lifted verbatim from RAGApp_hybrid.ipynb.

Copied rather than rewritten so scores stay comparable with the linear baseline: the judge
prompts, the json_schema field order, the _grade fallback parser and the abstention
short-circuit all behave identically. The only change is that the judge model is
constructed through _judge_model() instead of at import time.
"""

import os
from langchain.chat_models import init_chat_model

JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "gemma4:12b")


def _judge_model():
    return init_chat_model(JUDGE_MODEL, model_provider="ollama", temperature=0)


from typing_extensions import Annotated,TypedDict,Literal
from langchain.chat_models import init_chat_model
import re

# --- Judge model: gemma4:12b, running LOCALLY on Ollama ---

# --- Why NOT llama3.1:8b (the previous judge) ---
# It produced systematically false-negative verdicts on correctness and retrieval_relevance.
# The cause was NOT bad parsing — parsing_error was None every time. It was constrained
# decoding interacting badly with a small model: method="json_schema" forces the fields in
# schema order (explanation, then verdict), so llama3.1:8b wrote ONE throat-clearing sentence
# into `explanation` ("To determine the relevance... I will analyze each fact individually."),
# the grammar closed the string, and it had to emit a true/false token having reasoned about
# nothing. Forced to answer cold it defaulted to false — deterministically, 3/3 at temperature 0.
# The giveaway was self-contradiction: runs scored groundedness=1.0 (the answer IS supported by
# these chunks) AND retrieval_relevance=0.0 (the chunks are unrelated to the question) on the
# same chunks. gemma4:12b has the headroom to actually reason inside the `explanation` field
# before committing to a verdict, which is what the explanation-before-answer ordering assumes.
#
#   method="json_schema" -> Ollama constrained decoding forces a clean true/false token.
#   include_raw=True      -> .invoke() returns {raw, parsed, ...} instead of raising, so a
#                            malformed judge response degrades gracefully (see _grade()).
JUDGE_MODEL = "gemma4:12b"
model = _judge_model()

def _to_bool(v) -> bool:
    return str(v).strip().strip('.').strip().lower() in ("true", "yes", "1", "y", "t")

def _extract_bool(text: str, key: str) -> bool:
    """Fallback verdict parser: pull a true/false out of the judge's raw text when structured
    parsing fails, so a malformed judge response degrades gracefully instead of crashing."""
    if not text:
        return False
    m = re.search(rf'"?{key}"?\s*[:=]\s*"?(true|false|yes|no)"?', text, re.I)
    if m:
        return _to_bool(m.group(1))
    hits = re.findall(r'\b(true|false)\b', text, re.I)
    return _to_bool(hits[-1]) if hits else False

def _grade(grader, messages, key: str) -> bool:
    """Invoke a structured-output grader built with include_raw=True. That makes .invoke()
    return {"raw", "parsed", "parsing_error"} instead of raising on a bad parse — we use the
    parsed verdict when present, else fall back to scanning the raw text. This is the fix for
    the OutputParserException that previously produced None scores.

    NOTE: this fallback only catches PARSE failures. It cannot catch a judge that parses
    cleanly but reasons badly — that was the llama3.1:8b failure described above, and the
    only fix for it is a more capable judge model."""
    res = grader.invoke(messages)
    parsed = res.get("parsed") if isinstance(res, dict) else res
    if parsed:
        return _to_bool(parsed[key])
    raw = res.get("raw") if isinstance(res, dict) else None
    raw_text = getattr(raw, "text", "") or (getattr(raw, "content", "") if raw else "")
    return _extract_bool(str(raw_text), key)

# Correctness: Response vs reference answer
# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[Literal['true','false'], ..., "True if the answer is correct, False otherwise."]

## correctness prompt

correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""


# --- Shared abstention rule for the four answer-quality metrics -----------------------
# When the bot abstains there is no answer to grade, so these metrics would otherwise
# score a REFUSAL as if it were a bad answer — which is how a correct refusal came to be
# recorded as groundedness 0.25 / relevance 0.50 / retrieval_relevance 0.00.
#
# Instead, score the DECISION on those rows:
#   * unanswerable question + abstained -> True   (correctly declined; nothing to grade)
#   * answerable   question + abstained -> False  (over-refusal; the floor is too high)
# Answered rows fall through and are graded normally by the LLM judge.
#
# Returning a verdict rather than None keeps every metric defined on all 14 rows, so the
# means stay comparable across experiments. Note the consequence: on refusal rows all four
# metrics agree by construction, so they carry no independent signal there — they read as
# "did the system do the right thing", not "how good was the answer text".
def _abstention_verdict(outputs: dict, reference_outputs: dict):
    """Return True/False for an abstained row, or None if the bot actually answered."""
    if not bool(outputs.get("abstained", False)):
        return None
    return not bool(reference_outputs.get("answerable", True))


grader_llm=model.with_structured_output(CorrectnessGrade, method="json_schema", include_raw=True)
## evaluator
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    _v = _abstention_verdict(outputs, reference_outputs)
    if _v is not None:
        return _v          # a correct refusal IS the correct output for this question
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    return _grade(grader_llm, [
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ], "correct")

# Relevance: Response vs input
# The flow is similar to above, but we simply look at the inputs and outputs without needing the reference_outputs. 
# Without a reference answer we can't grade accuracy, but can still grade relevance—as in, did the model address the user's question or not.
# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[Literal['true','false'], ..., "Provide the score on whether the answer addresses the question"]

# Grade prompt
relevance_instructions="""You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = model.with_structured_output(RelevanceGrade, method="json_schema", include_raw=True)

# Evaluator
def relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    _v = _abstention_verdict(outputs, reference_outputs)
    if _v is not None:
        return _v          # declining an unanswerable question IS the helpful response
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    return _grade(relevance_llm, [
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ], "relevant")


# Groundedness: Response vs retrieved docs
# Another useful way to evaluate responses without needing reference answers is to check if the response is justified by (or "grounded in") the retrieved documents.
# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[Literal['true','false'], ..., "Provide the score on if the answer hallucinates from the documents"]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM 
grounded_llm = model.with_structured_output(GroundedGrade, method="json_schema", include_raw=True)

# Evaluator
def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    # An abstention asserts nothing, so it cannot be ungrounded — but an over-refusal on
    # an answerable question is still a failure, so score the decision rather than
    # returning True unconditionally.
    _v = _abstention_verdict(outputs, reference_outputs)
    if _v is not None:
        return _v
    doc_string = "\n\n".join(doc["content"] for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    return _grade(grounded_llm, [
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer}
    ], "grounded")


# Retrieval Relevance: Retrieved docs vs input
# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[Literal['true','false'], ..., "True if the retrieved documents are relevant to the question, False otherwise"]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = model.with_structured_output(RetrievalRelevanceGrade, method="json_schema", include_raw=True)

def retrieval_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for document relevance"""
    # Abstention means nothing cleared the floor. On an unanswerable question that is the
    # CORRECT retrieval outcome, so it scores True; on an answerable one it is a miss.
    _v = _abstention_verdict(outputs, reference_outputs)
    if _v is not None:
        return _v
    doc_string = "\n\n".join(doc["content"] for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    # Run evaluator
    return _grade(retrieval_relevance_llm, [
        {"role": "system", "content": retrieval_relevance_instructions}, 
        {"role": "user", "content": answer}
    ], "relevant")


# Abstention: did the bot refuse exactly when it should have?
# -----------------------------------------------------------------------------------
# No LLM involved — this is a label comparison, so it is deterministic, free, and cannot
# be swayed by a persuasive wrong answer. It is the ONLY metric that scores the refusal
# path, and it reads in both directions:
#   * unanswerable question + abstained    -> True   (correctly declined)
#   * unanswerable question + answered     -> False  (hallucinated; the failure that matters)
#   * answerable question   + answered     -> True   (correctly proceeded)
#   * answerable question   + abstained    -> False  (over-refusal; the floor is too high)
# That last row is why the metric is not simply "refusal rate": a floor set high enough to
# refuse everything would score perfectly on a refusal-only measure.
def abstention(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Did the bot's decision to answer-or-refuse match what the question warranted?"""
    answerable = bool(reference_outputs.get("answerable", True))
    abstained  = bool(outputs.get("abstained", False))
    return abstained != answerable


# The five metrics this single judge scores. Defined here, beside the evaluators
# themselves, so the evaluation cell below has no hidden ordering dependency.
evaluators = [correctness, groundedness, relevance, retrieval_relevance, abstention]
print("judge:", f"{JUDGE_MODEL} (local ollama)", "| evaluators:", [e.__name__ for e in evaluators])
