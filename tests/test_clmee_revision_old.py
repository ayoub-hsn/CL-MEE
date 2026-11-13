import unittest
import numpy as np

from run_revision_experiments import (
    k_from_deltas,
    make_evidence,
    evaluate_budget_matched_posthoc,
    evaluate_policy_full,
    metric_dict,
    load_weight_checkpoint,
    sha256_file,
)


class TestCLMERevision(unittest.TestCase):
    def test_zero_evidence_has_zero_display_not_fictitious_reasons(self):
        deltas = np.zeros((2, 3, 5), dtype=float)
        K, sorted_delta, totals, cumulative = k_from_deltas(deltas, rho=0.80)
        self.assertTrue(np.all(K == 5))  # conservative ranking-burden surrogate
        self.assertTrue(np.all(totals == 0))
        # The display rule is tested through the post-hoc policy below.
        features = np.ones((2, 3, 5), dtype=np.float32)
        weights = np.ones(5) / 5
        evidence = make_evidence(features, weights)
        target_ids = np.array([10, 20])
        candidate_ids = np.array([[10, 11, 12], [20, 21, 22]])
        base_ranks = np.array([1, 2])
        scores = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
        post = evaluate_budget_matched_posthoc(
            scores, candidate_ids, target_ids, base_ranks, evidence, B=1
        )
        self.assertTrue(np.all(post["displayed_L"] == 0))

    def test_budget_matched_posthoc_preserves_ranking_and_respects_budget(self):
        features = np.array([
            [[0.9, 0.2, 0.3, 0.1, 0.2], [0.7, 0.3, 0.2, 0.1, 0.1], [0.2, 0.2, 0.4, 0.1, 0.1]],
            [[0.2, 0.8, 0.4, 0.1, 0.2], [0.3, 0.7, 0.2, 0.1, 0.1], [0.1, 0.2, 0.5, 0.1, 0.1]],
        ], dtype=np.float32)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        evidence = make_evidence(features, weights)
        candidate_ids = np.array([[10, 11, 12], [20, 21, 22]])
        target_ids = np.array([12, 20])
        base_ranks = np.array([3, 1])
        scores = np.array([[0.8, 0.7, 0.6], [0.9, 0.8, 0.7]])
        post = evaluate_budget_matched_posthoc(
            scores, candidate_ids, target_ids, base_ranks, evidence, B=1
        )
        self.assertEqual(post["metrics"], metric_dict(base_ranks))
        self.assertEqual(post["rerank_rate"], 0.0)
        self.assertEqual(post["budget_violations_after_truncation"], 0)
        self.assertTrue(np.all(post["displayed_L"] <= 1))

    def test_common_runner_up_anchor_uses_same_reference_for_each_candidate(self):
        features = np.array([[[0.8, 0.2, 0.1], [0.4, 0.5, 0.1], [0.2, 0.3, 0.7]]], dtype=np.float32)
        weights = np.array([0.5, 0.3, 0.2])
        ev = make_evidence(features, weights, anchor="runner_up_common")
        contributions = features.astype(float) * weights[None, None, :]
        expected = np.maximum(0, contributions - contributions[:, 1:2, :])
        np.testing.assert_allclose(ev["delta"], expected)
        self.assertTrue(np.all(ev["delta"][:, 1, :] == 0))

    def test_re_ranking_respects_score_envelope_and_display_budget(self):
        features = np.array([[[0.9, 0.1, 0.2], [0.2, 0.9, 0.1], [0.5, 0.3, 0.2]]], dtype=np.float32)
        weights = np.array([0.4, 0.4, 0.2])
        evidence = make_evidence(features, weights)
        top_scores = np.array([[0.90, 0.87, 0.82]])
        candidate_ids = np.array([[100, 101, 102]])
        target_ids = np.array([102])
        base_ranks = np.array([5])
        result = evaluate_policy_full(
            top_scores, candidate_ids, target_ids, base_ranks, evidence,
            B=1, lam=1.0, epsilon=0.05, eta=0.008
        )
        self.assertLessEqual(result["score_loss_all"]["max"], 0.05 + 1e-10)
        self.assertEqual(result["budget_violations_after_truncation"], 0)
        self.assertTrue(np.all(result["displayed_L"] <= 1))

    def test_identity_control_keeps_base_order_even_with_ties(self):
        features = np.array([[[0.2, 0.4, 0.1], [0.4, 0.2, 0.1], [0.1, 0.1, 0.5]]], dtype=np.float32)
        weights = np.array([0.4, 0.4, 0.2])
        evidence = make_evidence(features, weights)
        top_scores = np.array([[0.8, 0.8, 0.7]])
        candidate_ids = np.array([[10, 11, 12]])
        target_ids = np.array([12])
        base_ranks = np.array([3])
        result = evaluate_policy_full(
            top_scores, candidate_ids, target_ids, base_ranks, evidence,
            B=3, lam=0.0, epsilon=0.0, identity_control=True
        )
        np.testing.assert_array_equal(result["ranks"], base_ranks)
        self.assertEqual(result["rerank_rate"], 0.0)

    def test_validation_weight_checkpoint_matches_dataset_and_is_normalized(self):
        root = __import__("pathlib").Path(__file__).resolve().parents[1]
        dataset = root / "dataset_TSMC2014_NYC.txt"
        checkpoint = root / "foursquare_model_weights.json"
        weights, info = load_weight_checkpoint(checkpoint, sha256_file(dataset))
        self.assertEqual(weights.shape, (5,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(weights >= 0))
        self.assertEqual(info["weight_source"], "stored validation-trained checkpoint")


if __name__ == "__main__":
    unittest.main(verbosity=2)
