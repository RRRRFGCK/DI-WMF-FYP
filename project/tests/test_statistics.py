import unittest

from analyze_multiple_comparisons import holm_adjust


class MultipleComparisonTests(unittest.TestCase):
    def test_holm_adjustment_matches_known_example(self):
        adjusted = holm_adjust([0.01, 0.02, 0.04])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)

    def test_holm_adjustment_preserves_input_order_and_bounds(self):
        adjusted = holm_adjust([0.8, 0.001, 0.2, 0.03])
        self.assertEqual(len(adjusted), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertAlmostEqual(adjusted[1], 0.004)
        self.assertGreaterEqual(adjusted[0], adjusted[2])


if __name__ == "__main__":
    unittest.main()
