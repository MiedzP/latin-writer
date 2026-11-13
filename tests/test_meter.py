"""Tests for the heuristic metrical checker. Run: python -m unittest discover -s tests -v"""
import unittest

from latinwriter.meter import scan, scan_couplet


class HexameterTests(unittest.TestCase):
    # Known hexameters (Aeneid 1.1, Eclogues 1.1-2, Aeneid 1.2).
    PASS = [
        "Arma virumque cano Troiae qui primus ab oris",
        "Tityre tu patulae recubans sub tegmine fagi",
        "Ille ego qui quondam gracili modulatus avena",
        "Italiam fato profugus Laviniaque venit",
    ]
    FAIL_SYLLABLES = [
        "Iam uenit ad urbem",                                  # 6 syllables
        "Lorem ipsum dolor sit amet consectetur adipiscing",   # English, 16 syllables, no feet
    ]

    def test_known_hexameters_pass(self):
        for line in self.PASS:
            with self.subTest(line=line):
                r = scan(line, "hexameter")
                self.assertTrue(r.ok, r.reason)
                self.assertEqual(len(r.pattern), r.syllables)

    def test_bad_lines_fail(self):
        for line in self.FAIL_SYLLABLES:
            with self.subTest(line=line):
                self.assertFalse(scan(line, "hexameter").ok)

    def test_elision_drops_final_vowel_before_vowel(self):
        from latinwriter.meter import weigh
        # "multa in urbe": the final -a of multa is elided before "in", leaving mult-in-ur-be.
        weights, _ = weigh("multa in urbe")
        self.assertEqual(len(weights), 4)

    def test_diphthong_is_heavy(self):
        from latinwriter.meter import weigh
        weights, _ = weigh("aequ")
        self.assertEqual(weights[0], "H")


class CoupletTests(unittest.TestCase):
    def test_couplet_checks_alternate_forms(self):
        results = scan_couplet([
            "Arma virumque cano Troiae qui primus ab oris",   # hexameter line
            "Arma virumque cano Troiae qui primus ab oris",   # wrong: 15 syllables for a pentameter
        ])
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)
        self.assertIn("pentameter has 14", results[1].reason)


class ProseTests(unittest.TestCase):
    def test_prose_is_never_metre_checked(self):
        self.assertTrue(scan("Gallia est omnis divisa in partes tres", "prose").ok)


if __name__ == "__main__":
    unittest.main()
