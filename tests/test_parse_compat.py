import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bluecad_cost.scoring import parse_prediction  # noqa: E402


class ParseCompatTest(unittest.TestCase):
    """Compatibility cases against the official benchmark's normalize()."""

    def test_cot_markdown_verdict(self):
        self.assertEqual(parse_prediction("Reasoning...\n**Yes**", 1, cot=True), "yes")
        self.assertEqual(parse_prediction("Steps...\nFinal answer: No", 1, cot=True), "no")

    def test_cot_sentence_containing_no_is_unparseable(self):
        self.assertIsNone(parse_prediction("Analysis\nthere is no match here", 1, cot=True))

    def test_task2_lowercase_letter(self):
        self.assertEqual(parse_prediction("the answer is c", 2), "C")
        self.assertEqual(parse_prediction("Answer: (b)", 2), "B")

    def test_task2_pronoun_i_is_not_a_label(self):
        self.assertIsNone(parse_prediction("I think it matches none of them", 2))


if __name__ == "__main__":
    unittest.main()
