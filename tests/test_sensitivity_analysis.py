import unittest

import run_revision_experiments as clmee


class TestSensitivityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caches = [clmee.generate_seed_cache(seed) for seed in clmee.SIM_SEEDS]
        cls.result = clmee.run_sensitivity_analysis(cls.caches)

    def test_expected_five_factors_and_fifteen_settings(self):
        self.assertEqual(len(self.result["summary"]), 15)
        self.assertEqual(len(self.result["seed_level"]), 150)
        self.assertEqual(
            {row["factor"] for row in self.result["summary"]},
            {
                "evidence_coverage_rho",
                "penalty_scale",
                "budget_policy",
                "utility_envelope_scale",
                "score_noise_sigma",
            },
        )

    def test_each_setting_uses_ten_fixed_seeds(self):
        for row in self.result["summary"]:
            self.assertEqual(row["seeds"], 10)
        for factor in clmee.SENSITIVITY_VALUES:
            rows = [r for r in self.result["seed_level"] if r["factor"] == factor]
            self.assertEqual(len(rows), 30)
            self.assertEqual({r["seed"] for r in rows}, set(clmee.SIM_SEEDS))

    def test_sensitivity_ranges_match_manuscript_rounding(self):
        expected = {
            "evidence_coverage_rho": (99.54, 99.82),
            "penalty_scale": (99.62, 99.92),
            "budget_policy": (99.59, 100.02),
            "utility_envelope_scale": (99.68, 99.89),
            "score_noise_sigma": (99.66, 99.70),
        }
        ranges = self.result["ranges_across_setting_means"]
        for factor, (expected_min, expected_max) in expected.items():
            self.assertEqual(round(ranges[factor]["NDCG_retained_pct_min"], 2), expected_min)
            self.assertEqual(round(ranges[factor]["NDCG_retained_pct_max"], 2), expected_max)

    def test_primary_central_setting_is_present(self):
        central = [
            r for r in self.result["summary"]
            if (r["factor"], r["setting"]) in {
                ("evidence_coverage_rho", "0.80"),
                ("penalty_scale", "1.0x"),
                ("budget_policy", "3/2/1"),
                ("utility_envelope_scale", "1.0x"),
                ("score_noise_sigma", "0.015"),
            }
        ]
        self.assertEqual(len(central), 5)
        for row in central:
            self.assertAlmostEqual(row["NDCG_retained_pct_mean"], 99.6780, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
