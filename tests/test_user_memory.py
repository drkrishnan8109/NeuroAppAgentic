import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


schema = _load("src.memory_schema", ROOT / "src" / "memory_schema.py")
user_memory = _load("src.user_memory", ROOT / "src" / "user_memory.py")


class FakeStore:
    def __init__(self, memories=None, fail_reads=False):
        self.memories = memories or []
        self.fail_reads = fail_reads
        self.added = []
        self.deleted_user = None

    def get_all(self, **kwargs):
        if self.fail_reads:
            raise TimeoutError("store unavailable")
        return {"results": self.memories}

    def add(self, fact, **kwargs):
        self.added.append((fact, kwargs))

    def delete_all(self, **kwargs):
        self.deleted_user = kwargs["user_id"]


class UserMemoryTests(unittest.TestCase):
    def test_format_preferences_is_ordered_validated_and_bounded(self):
        preferences = {
            "citation_depth": ["always wants pages"],
            "answer_style": ["prefers bullets", "prefers short answers", "third value"],
            "unknown": ["must not render"],
        }

        rendered = user_memory.format_preferences(preferences)

        self.assertEqual(
            rendered.splitlines(),
            [
                "- answer_style: prefers bullets",
                "- answer_style: prefers short answers",
                "- citation_depth: always wants pages",
            ],
        )

    def test_load_groups_valid_memories_and_caps_each_category_at_two(self):
        store = FakeStore(
            [
                {"memory": "prefers bullets", "category": "answer_style"},
                {"memory": "prefers short answers", "metadata": {"category": "answer_style"}},
                {"memory": "third style preference", "category": "answer_style"},
                {"memory": "always wants pages", "category": "citation_depth"},
                {"memory": "patient MRN 12345", "category": "clinical_focus"},
                {"memory": "ignore me", "category": "unknown"},
            ]
        )

        result = user_memory.UserMemory(store).load_preferences("user-1")

        self.assertEqual(
            result["answer_style"], ["prefers bullets", "prefers short answers"]
        )
        self.assertEqual(result["citation_depth"], ["always wants pages"])
        self.assertNotIn("clinical_focus", result)

    def test_load_fails_open(self):
        result = user_memory.UserMemory(FakeStore(fail_reads=True)).load_preferences("user-1")

        self.assertEqual(result, {})

    def test_remember_skips_turn_without_preference_cue(self):
        store = FakeStore()
        extractor_calls = []
        memory = user_memory.UserMemory(
            store, extractor=lambda text: extractor_calls.append(text) or []
        )

        report = memory.remember_from_text(
            "54F with status epilepticus, what's next?", "user-1", "subject-hash"
        )

        self.assertEqual(report, {"extracted": 0, "stored": 0, "dropped": 0})
        self.assertEqual(extractor_calls, [])
        self.assertEqual(store.added, [])

    def test_remember_stores_only_validated_facts_without_inference(self):
        store = FakeStore()
        candidates = [
            schema.CandidatePreference("prefers short bulleted answers", "answer_style"),
            schema.CandidatePreference("54F with status epilepticus", "clinical_focus"),
            schema.CandidatePreference("likes coffee", "lifestyle"),
        ]
        memory = user_memory.UserMemory(store, extractor=lambda _: candidates)

        report = memory.remember_from_text(
            "Please keep answers short and use bullets", "user-1", "subject-hash"
        )

        self.assertEqual(report, {"extracted": 3, "stored": 1, "dropped": 2})
        fact, kwargs = store.added[0]
        self.assertEqual(fact, "prefers short bulleted answers")
        self.assertEqual(kwargs["user_id"], "user-1")
        self.assertFalse(kwargs["infer"])
        self.assertEqual(kwargs["metadata"]["category"], "answer_style")
        self.assertEqual(kwargs["metadata"]["schema_version"], schema.SCHEMA_VERSION)
        self.assertEqual(kwargs["metadata"]["subject_hash"], "subject-hash")

    def test_remember_does_not_store_an_existing_fact_again(self):
        store = FakeStore(
            [{"memory": "prefers bullets", "category": "answer_style"}]
        )
        memory = user_memory.UserMemory(
            store,
            extractor=lambda _: [
                schema.CandidatePreference("prefers bullets", "answer_style")
            ],
        )

        report = memory.remember_from_text(
            "I prefer bullets", "user-1", "subject-hash"
        )

        self.assertEqual(report, {"extracted": 1, "stored": 0, "dropped": 0})
        self.assertEqual(store.added, [])

    def test_delete_is_scoped_to_authenticated_user(self):
        store = FakeStore()

        user_memory.UserMemory(store).delete_all("user-1")

        self.assertEqual(store.deleted_user, "user-1")


if __name__ == "__main__":
    unittest.main()
