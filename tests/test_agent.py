"""Loop tests with the model replaced by scripted replies. Run: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from latinwriter import agent

GOOD_VERSE = (
    "LATIN:\nArma virumque cano Troiae qui primus ab oris\n"
    "TRANSLATION:\nI sing of arms and the man, who first came from the shores of Troy\n"
)
SHORT_VERSE = (
    "LATIN:\nIam uenit ad urbem\nTRANSLATION:\nHe now comes to the city\n"
)


class AgentLoopTests(unittest.TestCase):
    def test_metre_failure_then_fix(self):
        replies = iter([SHORT_VERSE, GOOD_VERSE])
        judgments = iter([
            {"score": 70, "errors": [], "strengths": []},
            {"score": 90, "errors": [], "strengths": ["clear"]},
        ])
        with mock.patch.object(agent, "ask", side_effect=lambda *a, **k: next(replies)), \
             mock.patch.object(agent, "_judge", side_effect=lambda *a, **k: next(judgments)):
            out = Path(tempfile.mkdtemp(prefix="lw_test_"))
            report = agent.write(form="hexameter", style="vergil", theme="the Aeneid opening",
                                 out=out, lines=1, max_iters=4, threshold=80, log=lambda *_: None)

        self.assertEqual(report["result"], "ok")
        self.assertEqual(report["best_iteration"], 1)
        self.assertEqual(report["attempts"][0]["meter_failures"], 1)
        self.assertEqual(report["attempts"][0]["score"], 60)   # 70 minus one metre penalty
        self.assertEqual(report["attempts"][1]["score"], 90)
        piece = (out / "piece.txt").read_text(encoding="utf-8")
        self.assertIn("Arma virumque", piece)

    def test_judge_errors_reach_the_generator(self):
        seen = []

        def fake_ask(messages, system, model=None, max_tokens=None, json_mode=False):
            seen.append(messages[-1]["content"])
            return GOOD_VERSE

        judgments = iter([
            {"score": 50, "errors": [{"line": 1, "problem": "wrong case", "fix": "oris"}]},
            {"score": 95, "errors": []},
        ])
        with mock.patch.object(agent, "ask", side_effect=fake_ask), \
             mock.patch.object(agent, "_judge", side_effect=lambda *a, **k: next(judgments)):
            agent.write(form="hexameter", style="vergil", theme="x",
                        out=Path(tempfile.mkdtemp()), max_iters=3, log=lambda *_: None)
        self.assertIn("wrong case", seen[-1])

    def test_below_threshold_when_never_passes(self):
        with mock.patch.object(agent, "ask", return_value=SHORT_VERSE), \
             mock.patch.object(agent, "_judge", return_value={"score": 99, "errors": []}):
            report = agent.write(form="hexameter", style="vergil", theme="x",
                                 out=Path(tempfile.mkdtemp()), max_iters=2, log=lambda *_: None)
        self.assertEqual(report["result"], "below_threshold")
        self.assertEqual(len(report["attempts"]), 2)

    def test_prose_skips_metre(self):
        prose = "LATIN:\nGallia est omnis divisa in partes tres.\nTRANSLATION:\nAll Gaul is divided into three parts.\n"
        with mock.patch.object(agent, "ask", return_value=prose), \
             mock.patch.object(agent, "_judge", return_value={"score": 85, "errors": []}):
            report = agent.write(form="prose", style="caesar", theme="Gaul",
                                 out=Path(tempfile.mkdtemp()), max_iters=1, log=lambda *_: None)
        self.assertEqual(report["result"], "ok")
        self.assertEqual(report["attempts"][0]["meter"], [])


if __name__ == "__main__":
    unittest.main()
