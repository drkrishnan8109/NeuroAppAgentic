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
runtime = _load("src.memory_runtime", ROOT / "src" / "memory_runtime.py")


class FakeStructuredLlm:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.result


class MemoryRuntimeTests(unittest.TestCase):
    def test_extractor_converts_structured_output_to_candidates(self):
        llm = FakeStructuredLlm(
            {
                "preferences": [
                    {"fact": "prefers bullets", "category": "answer_style"},
                    {"fact": "always wants pages", "category": "citation_depth"},
                ]
            }
        )

        result = runtime.make_preference_extractor(llm)("Please use bullets and pages")

        self.assertEqual(
            result,
            [
                schema.CandidatePreference("prefers bullets", "answer_style"),
                schema.CandidatePreference("always wants pages", "citation_depth"),
            ],
        )
        self.assertEqual(llm.messages[0]["content"], schema.CUSTOM_INSTRUCTIONS)

    def test_extractor_ignores_malformed_rows(self):
        llm = FakeStructuredLlm(
            {"preferences": [{"fact": "prefers bullets"}, "bad", None]}
        )

        result = runtime.make_preference_extractor(llm)("Please use bullets")

        self.assertEqual(result, [])

    def test_free_tier_config_matches_embedding_dimensions(self):
        config = runtime.build_mem0_config("postgresql://example", "google-key")

        self.assertEqual(config["embedder"]["config"]["embedding_dims"], 768)
        self.assertEqual(
            config["vector_store"]["config"]["embedding_model_dims"], 768
        )
        self.assertEqual(config["vector_store"]["provider"], "supabase")


if __name__ == "__main__":
    unittest.main()
