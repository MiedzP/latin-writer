"""Two-pass judge tests with scripted model replies and a fake grammar index. No book text is read.
Run: python -m unittest discover -s tests -v"""
import json
import unittest
from unittest import mock

from latinwriter import judge


class FakeIndex:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=3):
        self.queries.append(query)
        return [{"page": 313, "part": "syntax of clauses", "text": "reference text about ut clauses"}]


def reply(score, confidence, query="purpose clause with ut"):
    return json.dumps({"score": score, "errors": [], "strengths": [],
                       "confidence": confidence, "reference_query": query})


class TwoPassJudgeTests(unittest.TestCase):
    def test_confident_first_pass_never_escalates(self):
        calls = []

        def fake_ask(messages, system, model=None, max_tokens=None):
            calls.append(system)
            return reply(90, "high")

        index = FakeIndex()
        with mock.patch.object(judge, "ask", side_effect=fake_ask):
            out = judge.judge_latin("Gallia est omnis divisa.", "prose", "caesar", "x", model="m", index=index)
        self.assertEqual(len(calls), 1)
        self.assertFalse(out["escalated"])
        self.assertEqual(index.queries, [])

    def test_low_confidence_escalates_with_reference(self):
        calls = []

        def fake_ask(messages, system, model=None, max_tokens=None):
            calls.append(system)
            return reply(60, "low") if len(calls) == 1 else reply(85, "high")

        index = FakeIndex()
        with mock.patch.object(judge, "ask", side_effect=fake_ask):
            out = judge.judge_latin("Cavet ut veniat.", "prose", "cicero", "x", model="m", index=index)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("Reference pages", calls[0])
        self.assertIn("Reference pages", calls[1])
        self.assertIn("reference text about ut clauses", calls[1])
        self.assertTrue(out["escalated"])
        self.assertEqual(out["reference_pages"], [313])
        self.assertEqual(out["first_pass_score"], 60)
        self.assertEqual(out["score"], 85)
        self.assertIn("purpose clause with ut", index.queries[0])

    def test_no_book_means_single_pass(self):
        with mock.patch.object(judge, "ask", return_value=reply(50, "low")), \
             mock.patch.object(judge, "_grammar_index", return_value=None):
            out = judge.judge_latin("x", "prose", "plain", "y", model="m")
        self.assertFalse(out["escalated"])
        self.assertIn("no grammar book", out["note"])


if __name__ == "__main__":
    unittest.main()
