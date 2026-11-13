#!/usr/bin/env python3
"""Re-run CL-MEE and reviewer-requested revision experiments.

This script is self-contained and does not rely on notebook state. It reproduces
 the supplied simulation/Foursquare protocol using the packaged validation-
 trained scorer checkpoint by default (or re-fits it when --refit-weights is
 specified), then adds:
  * a budget-matched post-hoc truncation baseline;
  * fixed-parameter budget ablation and lambda=0 ablation;
  * five one-factor-at-a-time simulation sensitivity analyses (rho, penalty
    scale, budget policy, utility-envelope width, and score noise);
  * a common-runner-up comparison-anchor sensitivity check;
  * operational diagnostics and runtime measurements;
  * a real-data positive/negative contribution example;
  * explicit event-weighted and user-weighted metric reporting.

Run from this directory:
    python run_revision_experiments.py --dataset dataset_TSMC2014_NYC.txt

The data are not downloaded by this script. It uses only the supplied dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import minimize
from sklearn.neighbors import BallTree
import sklearn

# ---------------------------------------------------------------------------
# Fixed protocol constants (preserved from the submitted notebook)
# ---------------------------------------------------------------------------
SIM_SEEDS = list(range(10))
SIM_USERS = 800
SIM_POIS = 600
SIM_CATEGORIES = 10
SIM_SESSIONS_PER_USER = 3
SIM_M = 40
SIM_RHO = 0.80
SIM_ETA = 0.008
SIM_CONTEXT_PROBS = (0.35, 0.40, 0.25)
SIM_BUDGET_POLICY = (3, 2, 1)  # detailed, standard, quick-glance modes
SIM_LAMBDAS = (0.0, 0.035, 0.075)
SIM_EPSILONS = (0.0, 0.05, 0.10)
SIM_LATENT_WEIGHTS = np.array([0.50, 0.35, 0.15], dtype=float)
SIM_MODEL_WEIGHTS = np.array([0.52, 0.31, 0.17], dtype=float)
SIM_LATENT_NOISE = 0.205
SIM_SCORE_NOISE = 0.015
SIM_DIRICHLET_ALPHA = 0.40
SIM_DISTANCE_TAU = 0.25

# The exact one-factor-at-a-time settings reported in the manuscript's
# sensitivity table. The central setting is included in every sweep.
SENSITIVITY_VALUES = {
    "evidence_coverage_rho": (0.70, 0.80, 0.90),
    "penalty_scale": (0.5, 1.0, 1.5),
    "budget_policy": ((1, 1, 1), (3, 2, 1), (3, 3, 3)),
    "utility_envelope_scale": (0.5, 1.0, 1.5),
    "score_noise_sigma": (0.010, 0.015, 0.030),
}

RANDOM_SEED = 20260919
N_NEAR = 249
N_POP = 250
N_CANDIDATES = 1 + N_NEAR + N_POP
TOP_M = 40
RHO = 0.80
ETA = 0.008
ALPHA = 0.10
FEATURE_NAMES = [
    "User preference",
    "Proximity",
    "Popularity",
    "Temporal compatibility",
    "Transition compatibility",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): safe_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_float(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def metric_dict(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=int)
    return {
        "Recall@5": float(np.mean(ranks <= 5)),
        "NDCG@5": float(np.mean(np.where(ranks <= 5, 1.0 / np.log2(ranks + 1), 0.0))),
        "MRR": float(np.mean(1.0 / ranks)),
    }


def ndcg_contribution(ranks: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=int)
    return np.where(ranks <= 5, 1.0 / np.log2(ranks + 1), 0.0)


def paired_mean_ci(x: np.ndarray, y: np.ndarray, confidence: float = 0.95):
    diff = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    if n < 2:
        return mean, [None, None]
    se = stats.sem(diff)
    crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, [float(mean - crit * se), float(mean + crit * se)]


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in ["mean", "median", "p90", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


# ---------------------------------------------------------------------------
# Simulation protocol (preserved; adds budget-matched post-hoc baseline)
# ---------------------------------------------------------------------------
def generate_seed_cache(seed: int, latent_sigma=SIM_LATENT_NOISE,
                        score_sigma=SIM_SCORE_NOISE, alpha=SIM_DIRICHLET_ALPHA,
                        tau=SIM_DISTANCE_TAU, M=SIM_M):
    rng = np.random.default_rng(seed)
    U, P, C, S = SIM_USERS, SIM_POIS, SIM_CATEGORIES, SIM_SESSIONS_PER_USER
    poi_cat = rng.integers(C, size=P)
    poi_xy = rng.random((P, 2))
    popularity = rng.beta(2, 5, size=P)
    popularity = popularity / popularity.max()
    user_preferences = rng.dirichlet(np.full(C, alpha), size=U)
    user_centres = rng.random((U, 2))

    n_sessions = U * S
    base_ranks = np.empty(n_sessions, dtype=np.int16)
    top_scores = np.empty((n_sessions, M), dtype=np.float32)
    top_is_target = np.zeros((n_sessions, M), dtype=bool)
    deltas = np.empty((n_sessions, M, 3), dtype=np.float32)
    contexts = np.empty(n_sessions, dtype=np.int8)

    row = 0
    for u in range(U):
        for _ in range(S):
            contexts[row] = rng.choice(3, p=SIM_CONTEXT_PROBS)
            current_xy = np.clip(user_centres[u] + rng.normal(0, 0.15, size=2), 0, 1)
            distance = np.linalg.norm(poi_xy - current_xy, axis=1)
            proximity = np.exp(-distance / tau)
            preference = user_preferences[u, poi_cat]
            X = np.column_stack([preference, proximity, popularity])
            latent_utility = X @ SIM_LATENT_WEIGHTS + rng.normal(0, latent_sigma, size=P)
            target = int(np.argmax(latent_utility))
            contributions = X * SIM_MODEL_WEIGHTS[None, :]
            scores = contributions.sum(axis=1) + rng.normal(0, score_sigma, size=P)
            order = np.argsort(-scores, kind="stable")
            base_ranks[row] = int(np.where(order == target)[0][0]) + 1
            top = order[:M]
            top_scores[row] = scores[top]
            top_is_target[row] = top == target
            deltas[row, 0] = np.maximum(0.0, contributions[top[0]] - contributions[top[1]])
            deltas[row, 1:] = np.maximum(0.0, contributions[top[1:]] - contributions[top[0]])
            row += 1
    return {"base_ranks": base_ranks, "top_scores": top_scores,
            "top_is_target": top_is_target, "deltas": deltas, "contexts": contexts}


def k_from_deltas(deltas: np.ndarray, rho: float):
    totals = deltas.sum(axis=2)
    sorted_delta = np.sort(deltas, axis=2)[:, :, ::-1]
    cumulative = np.cumsum(sorted_delta, axis=2)
    m = deltas.shape[2]
    K = np.full(totals.shape, m, dtype=np.int8)
    positive = totals > 1e-12
    ratios = np.zeros_like(cumulative, dtype=np.float32)
    ratios[positive] = cumulative[positive] / totals[positive, None]
    first = np.argmax(ratios >= rho, axis=2) + 1
    K[positive] = first[positive]
    return K, sorted_delta, totals, cumulative


def evaluate_simulation_cache(cache: dict, rho=SIM_RHO,
                              penalty_scale=1.0,
                              budget_policy=SIM_BUDGET_POLICY,
                              envelope_scale=1.0, eta=SIM_ETA):
    base_ranks = cache["base_ranks"]
    scores = cache["top_scores"]
    is_target = cache["top_is_target"]
    contexts = cache["contexts"]
    deltas = cache["deltas"]
    K, sorted_delta, totals, cumulative = k_from_deltas(deltas, rho)
    n, n_candidates, n_features = deltas.shape
    B = np.asarray(budget_policy, dtype=int)[contexts]
    lam = np.asarray(SIM_LAMBDAS) [contexts] * penalty_scale
    eps = np.asarray(SIM_EPSILONS)[contexts] * envelope_scale
    eligible = scores >= (scores[:, [0]] - eps[:, None])
    burden = np.maximum(0, K - B[:, None])
    adjusted = scores - lam[:, None] * burden - eta * K / n_features
    adjusted = np.where(eligible, adjusted, -np.inf)
    chosen_pos = np.argmax(adjusted, axis=1)
    chosen_is_target = is_target[np.arange(n), chosen_pos]
    cl_rank = base_ranks.astype(int).copy()
    cl_rank[chosen_is_target] = 1
    shift = (~chosen_is_target) & ((chosen_pos + 1) > base_ranks)
    cl_rank[shift] += 1

    # Minimal-posthoc: no display cap.
    k0 = K[:, 0]
    d0 = totals[:, 0]
    pos0 = d0 > 1e-12
    post_units = np.where(pos0, k0, 0)
    post_num = cumulative[np.arange(n), 0, np.maximum(k0 - 1, 0)]
    post_f = np.divide(post_num, d0, out=np.zeros(n, dtype=float), where=pos0)

    # Budget-matched post-hoc: same positive-evidence prefix and hard truncation,
    # but no change to the base ranking.
    post_L = np.where(pos0, np.minimum(k0, B), 0)
    post_num_budget = np.where(post_L > 0,
                               cumulative[np.arange(n), 0, np.maximum(post_L - 1, 0)], 0.0)
    post_f_budget = np.divide(post_num_budget, d0, out=np.zeros(n, dtype=float), where=pos0)

    chosen_K = K[np.arange(n), chosen_pos]
    chosen_D = totals[np.arange(n), chosen_pos]
    chosen_L = np.where(chosen_D > 1e-12, np.minimum(chosen_K, B), 0)
    chosen_num = np.zeros(n, dtype=float)
    positive_display = chosen_L > 0
    chosen_num[positive_display] = cumulative[
        np.arange(n)[positive_display], chosen_pos[positive_display], chosen_L[positive_display] - 1
    ]
    cl_f = np.divide(chosen_num, chosen_D, out=np.zeros(n, dtype=float), where=chosen_D > 1e-12)

    return {
        "RankOnly": metric_dict(base_ranks),
        "CL-MEE": metric_dict(cl_rank),
        "Fixed-3": {"units": 3.0, "violations": float(np.mean(3 > B))},
        "Minimal-posthoc": {"units": float(post_units.mean()),
                            "violations": float(np.mean(post_units > B)),
                            "F_model": float(post_f.mean())},
        "Budget-matched post-hoc": {"metrics": metric_dict(base_ranks),
                                    "units": float(post_L.mean()),
                                    "violations": float(np.mean(post_L > B)),
                                    "F_model": float(post_f_budget.mean()),
                                    "rerank_rate": 0.0},
        "CL-MEE_explanation": {"units": float(chosen_L.mean()),
                               "violations": float(np.mean(chosen_L > B)),
                               "F_model": float(cl_f.mean()),
                               "rerank_rate": float(np.mean(chosen_pos != 0)),
                               "pre_truncation_overbudget": float(np.mean(chosen_K > B)),
                               "coverage_below_rho_positive_D": float(np.mean((chosen_D > 1e-12) & (cl_f < rho)))},
    }


def run_simulation() -> dict:
    caches = [generate_seed_cache(seed) for seed in SIM_SEEDS]
    runs = [evaluate_simulation_cache(c) for c in caches]
    methods = ["RankOnly", "Fixed-3", "Minimal-posthoc", "Budget-matched post-hoc", "CL-MEE"]
    rows = []
    for method in methods:
        if method == "RankOnly":
            src = [r["RankOnly"] for r in runs]
            rows.append({"Method": method,
                         "Hit@5_mean": np.mean([x["Recall@5"] for x in src]),
                         "NDCG@5_mean": np.mean([x["NDCG@5"] for x in src]),
                         "MRR_mean": np.mean([x["MRR"] for x in src]),
                         "Units": None, "Budget_violations": None, "F_model": None,
                         "Rerank_rate": 0.0})
        elif method == "Fixed-3":
            rows.append({"Method": method,
                         "Hit@5_mean": np.mean([r["RankOnly"]["Recall@5"] for r in runs]),
                         "NDCG@5_mean": np.mean([r["RankOnly"]["NDCG@5"] for r in runs]),
                         "MRR_mean": np.mean([r["RankOnly"]["MRR"] for r in runs]),
                         "Units": 3.0,
                         "Budget_violations": np.mean([r["Fixed-3"]["violations"] for r in runs]),
                         "F_model": None, "Rerank_rate": 0.0})
        elif method == "Minimal-posthoc":
            rows.append({"Method": method,
                         "Hit@5_mean": np.mean([r["RankOnly"]["Recall@5"] for r in runs]),
                         "NDCG@5_mean": np.mean([r["RankOnly"]["NDCG@5"] for r in runs]),
                         "MRR_mean": np.mean([r["RankOnly"]["MRR"] for r in runs]),
                         "Units": np.mean([r[method]["units"] for r in runs]),
                         "Budget_violations": np.mean([r[method]["violations"] for r in runs]),
                         "F_model": np.mean([r[method]["F_model"] for r in runs]),
                         "Rerank_rate": 0.0})
        elif method == "Budget-matched post-hoc":
            rows.append({"Method": method,
                         "Hit@5_mean": np.mean([r[method]["metrics"]["Recall@5"] for r in runs]),
                         "NDCG@5_mean": np.mean([r[method]["metrics"]["NDCG@5"] for r in runs]),
                         "MRR_mean": np.mean([r[method]["metrics"]["MRR"] for r in runs]),
                         "Units": np.mean([r[method]["units"] for r in runs]),
                         "Budget_violations": np.mean([r[method]["violations"] for r in runs]),
                         "F_model": np.mean([r[method]["F_model"] for r in runs]),
                         "Rerank_rate": 0.0})
        else:
            rows.append({"Method": method,
                         "Hit@5_mean": np.mean([r["CL-MEE"]["Recall@5"] for r in runs]),
                         "NDCG@5_mean": np.mean([r["CL-MEE"]["NDCG@5"] for r in runs]),
                         "MRR_mean": np.mean([r["CL-MEE"]["MRR"] for r in runs]),
                         "Units": np.mean([r["CL-MEE_explanation"]["units"] for r in runs]),
                         "Budget_violations": np.mean([r["CL-MEE_explanation"]["violations"] for r in runs]),
                         "F_model": np.mean([r["CL-MEE_explanation"]["F_model"] for r in runs]),
                         "Rerank_rate": np.mean([r["CL-MEE_explanation"]["rerank_rate"] for r in runs])})
    # Mean +/- SD over the ten independent seeds for the principal ranking metrics.
    for row in rows:
        method = row["Method"]
        if method in ("RankOnly", "Fixed-3", "Minimal-posthoc", "Budget-matched post-hoc"):
            vals = [r["RankOnly"] for r in runs]
        else:
            vals = [r["CL-MEE"] for r in runs]
        row["NDCG@5_SD"] = float(np.std([v["NDCG@5"] for v in vals], ddof=1))
        row["Hit@5_SD"] = float(np.std([v["Recall@5"] for v in vals], ddof=1))
        row["MRR_SD"] = float(np.std([v["MRR"] for v in vals], ddof=1))
    # Paired seed-level comparison against the budget-matched post-hoc baseline.
    cl_ndcg = np.asarray([r["CL-MEE"]["NDCG@5"] for r in runs], dtype=float)
    post_ndcg = np.asarray([r["Budget-matched post-hoc"]["metrics"]["NDCG@5"] for r in runs], dtype=float)
    cl_units = np.asarray([r["CL-MEE_explanation"]["units"] for r in runs], dtype=float)
    post_units = np.asarray([r["Budget-matched post-hoc"]["units"] for r in runs], dtype=float)
    cl_f = np.asarray([r["CL-MEE_explanation"]["F_model"] for r in runs], dtype=float)
    post_f = np.asarray([r["Budget-matched post-hoc"]["F_model"] for r in runs], dtype=float)
    paired_tests = {
        "NDCG_CL_MEE_minus_budget_matched": {"mean_difference": float((cl_ndcg-post_ndcg).mean()),
                                             "paired_t_p": float(stats.ttest_rel(cl_ndcg, post_ndcg).pvalue)},
        "units_CL_MEE_minus_budget_matched": {"mean_difference": float((cl_units-post_units).mean()),
                                              "paired_t_p": float(stats.ttest_rel(cl_units, post_units).pvalue)},
        "F_model_CL_MEE_minus_budget_matched": {"mean_difference": float((cl_f-post_f).mean()),
                                                "paired_t_p": float(stats.ttest_rel(cl_f, post_f).pvalue)},
    }
    sensitivity = run_sensitivity_analysis(caches)
    return {"table": rows, "seed_runs": runs, "paired_seed_tests": paired_tests,
            "sensitivity_analysis": sensitivity}


def run_sensitivity_analysis(baseline_caches: list[dict]) -> dict:
    """Run the five manuscript sensitivity sweeps, changing one factor at a time.

    The default caches are reused for rho, penalty, budget, and envelope sweeps.
    Score-noise settings regenerate the same ten seeds with only score_sigma
    changed. The primary simulation protocol and its reported results are not
    modified by these additional runs.
    """
    if len(baseline_caches) != len(SIM_SEEDS):
        raise ValueError("Sensitivity analysis requires one baseline cache per fixed seed")

    seed_level_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for factor, values in SENSITIVITY_VALUES.items():
        for value in values:
            if factor == "score_noise_sigma":
                if np.isclose(float(value), SIM_SCORE_NOISE):
                    caches = baseline_caches
                else:
                    caches = [generate_seed_cache(seed, score_sigma=float(value))
                              for seed in SIM_SEEDS]
            else:
                caches = baseline_caches

            kwargs = {
                "rho": SIM_RHO,
                "penalty_scale": 1.0,
                "budget_policy": SIM_BUDGET_POLICY,
                "envelope_scale": 1.0,
                "eta": SIM_ETA,
            }
            if factor == "evidence_coverage_rho":
                kwargs["rho"] = float(value)
                setting_label = f"{float(value):.2f}"
            elif factor == "penalty_scale":
                kwargs["penalty_scale"] = float(value)
                setting_label = f"{float(value):.1f}x"
            elif factor == "budget_policy":
                kwargs["budget_policy"] = tuple(int(x) for x in value)
                setting_label = "/".join(str(int(x)) for x in value)
            elif factor == "utility_envelope_scale":
                kwargs["envelope_scale"] = float(value)
                setting_label = f"{float(value):.1f}x"
            elif factor == "score_noise_sigma":
                setting_label = f"{float(value):.3f}"
            else:  # defensive guard if the declared grid is extended later
                raise ValueError(f"Unsupported sensitivity factor: {factor}")

            rows_for_setting = []
            for seed, cache in zip(SIM_SEEDS, caches):
                result = evaluate_simulation_cache(cache, **kwargs)
                baseline_ndcg = float(result["RankOnly"]["NDCG@5"])
                clmee_ndcg = float(result["CL-MEE"]["NDCG@5"])
                explanation = result["CL-MEE_explanation"]
                retained = (100.0 * clmee_ndcg / baseline_ndcg
                            if baseline_ndcg > 0 else float("nan"))
                row = {
                    "factor": factor,
                    "setting": setting_label,
                    "setting_value": ("/".join(str(int(x)) for x in value)
                                      if factor == "budget_policy" else float(value)),
                    "seed": int(seed),
                    "rho": float(kwargs["rho"]),
                    "penalty_scale": float(kwargs["penalty_scale"]),
                    "budget_policy": "/".join(str(int(x)) for x in kwargs["budget_policy"]),
                    "utility_envelope_scale": float(kwargs["envelope_scale"]),
                    "score_noise_sigma": (float(value) if factor == "score_noise_sigma"
                                           else float(SIM_SCORE_NOISE)),
                    "RankOnly_NDCG@5": baseline_ndcg,
                    "CL-MEE_NDCG@5": clmee_ndcg,
                    "NDCG_retained_pct": retained,
                    "F_model": float(explanation["F_model"]),
                    "displayed_units": float(explanation["units"]),
                    "rerank_rate": float(explanation["rerank_rate"]),
                    "budget_violations": float(explanation["violations"]),
                    "pre_truncation_overbudget_rate": float(explanation["pre_truncation_overbudget"]),
                    "coverage_below_rho_positive_D": float(explanation["coverage_below_rho_positive_D"]),
                }
                seed_level_rows.append(row)
                rows_for_setting.append(row)

            def mean_of(key: str) -> float:
                return float(np.mean([r[key] for r in rows_for_setting]))

            def sd_of(key: str) -> float:
                values_for_key = [r[key] for r in rows_for_setting]
                return float(np.std(values_for_key, ddof=1))

            mean_rankonly_ndcg = mean_of("RankOnly_NDCG@5")
            mean_clmee_ndcg = mean_of("CL-MEE_NDCG@5")
            # The manuscript reports retention as a ratio of seed-averaged
            # NDCG values (not the unweighted mean of ten per-seed ratios).
            retention_ratio_of_means = (100.0 * mean_clmee_ndcg / mean_rankonly_ndcg
                                        if mean_rankonly_ndcg > 0 else float("nan"))
            summary_rows.append({
                "factor": factor,
                "setting": setting_label,
                "setting_value": ("/".join(str(int(x)) for x in value)
                                  if factor == "budget_policy" else float(value)),
                "seeds": len(rows_for_setting),
                "NDCG_retained_pct_mean": retention_ratio_of_means,
                "NDCG_retained_pct_seed_mean": mean_of("NDCG_retained_pct"),
                "NDCG_retained_pct_seed_sd": sd_of("NDCG_retained_pct"),
                "RankOnly_NDCG@5_mean": mean_rankonly_ndcg,
                "CL-MEE_NDCG@5_mean": mean_clmee_ndcg,
                "F_model_mean": mean_of("F_model"),
                "F_model_sd": sd_of("F_model"),
                "displayed_units_mean": mean_of("displayed_units"),
                "displayed_units_sd": sd_of("displayed_units"),
                "rerank_rate_mean": mean_of("rerank_rate"),
                "budget_violations_mean": mean_of("budget_violations"),
                "pre_truncation_overbudget_rate_mean": mean_of("pre_truncation_overbudget_rate"),
                "coverage_below_rho_positive_D_mean": mean_of("coverage_below_rho_positive_D"),
            })

    # Machine-readable ranges are computed across setting-level means, exactly
    # as the manuscript's compact sensitivity table summarizes them.
    ranges = {}
    for factor in SENSITIVITY_VALUES:
        factor_rows = [r for r in summary_rows if r["factor"] == factor]
        ranges[factor] = {
            "NDCG_retained_pct_min": min(r["NDCG_retained_pct_mean"] for r in factor_rows),
            "NDCG_retained_pct_max": max(r["NDCG_retained_pct_mean"] for r in factor_rows),
            "F_model_min": min(r["F_model_mean"] for r in factor_rows),
            "F_model_max": max(r["F_model_mean"] for r in factor_rows),
            "displayed_units_min": min(r["displayed_units_mean"] for r in factor_rows),
            "displayed_units_max": max(r["displayed_units_mean"] for r in factor_rows),
        }

    return {
        "protocol": {
            "design": "one-factor-at-a-time; all other simulation parameters held at the central setting",
            "seeds": list(SIM_SEEDS),
            "sessions_per_seed": SIM_USERS * SIM_SESSIONS_PER_USER,
            "total_sessions_per_setting": SIM_USERS * SIM_SESSIONS_PER_USER * len(SIM_SEEDS),
            "central_rho": SIM_RHO,
            "central_penalty_scale": 1.0,
            "central_budget_policy": list(SIM_BUDGET_POLICY),
            "central_envelope_scale": 1.0,
            "central_score_noise_sigma": SIM_SCORE_NOISE,
            "lambda_by_context": list(SIM_LAMBDAS),
            "epsilon_by_context": list(SIM_EPSILONS),
            "eta": SIM_ETA,
            "tested_values": {
                key: [list(v) if isinstance(v, tuple) else v for v in values]
                for key, values in SENSITIVITY_VALUES.items()
            },
            "aggregation": "NDCG retention is 100 times the mean CL-MEE NDCG@5 across seeds divided by the mean RankOnly NDCG@5 across the same seeds, matching the manuscript table; per-seed ratios are retained in the raw CSV. Other reported metrics are means and sample SD across the ten fixed seeds.",
        },
        "summary": summary_rows,
        "seed_level": seed_level_rows,
        "ranges_across_setting_means": ranges,
    }


# ---------------------------------------------------------------------------
# Foursquare preprocessing, training and feature construction
# ---------------------------------------------------------------------------
def load_and_prepare(dataset_path: Path):
    columns = ["user_id", "venue_id", "category_id", "category_name",
               "latitude", "longitude", "timezone_offset", "utc_time"]
    df = pd.read_csv(dataset_path, sep="\t", names=columns, encoding="latin-1")
    df["utc_time"] = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y")
    df = df.sort_values(["user_id", "utc_time", "venue_id"], kind="stable").reset_index(drop=True)

    group_size = df.groupby("user_id")["user_id"].transform("size")
    group_position = df.groupby("user_id").cumcount()
    fraction = (group_position + 1) / group_size
    df["split"] = np.where(fraction <= 0.8, "train", np.where(fraction <= 0.9, "validation", "test"))

    df["prev_lat"] = df.groupby("user_id")["latitude"].shift(1)
    df["prev_lon"] = df.groupby("user_id")["longitude"].shift(1)
    df["prev_category"] = df.groupby("user_id")["category_name"].shift(1)
    df["local_time"] = df["utc_time"] + pd.to_timedelta(df["timezone_offset"], unit="m")
    df["prev_local_time"] = df.groupby("user_id")["local_time"].shift(1)

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    venue_meta = (train.groupby("venue_id", as_index=False)
                  .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                       category_name=("category_name", "first"), category_id=("category_id", "first"))
                  .reset_index(drop=True))
    venue_to_idx = pd.Series(venue_meta.index.values, index=venue_meta["venue_id"]).to_dict()
    train_venues = set(venue_meta["venue_id"])

    categories = pd.Index(sorted(train["category_name"].astype(str).unique()))
    category_to_idx = {category: i for i, category in enumerate(categories)}
    venue_meta["cat_idx"] = venue_meta["category_name"].astype(str).map(category_to_idx).astype(int)

    popularity = train.groupby("venue_id").size()
    venue_meta["pop_count"] = venue_meta["venue_id"].map(popularity).fillna(0).astype(int)
    venue_meta["popularity_feature"] = np.log1p(venue_meta["pop_count"]) / np.log1p(venue_meta["pop_count"].max())

    users = pd.Index(sorted(df["user_id"].unique()))
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users, n_categories = len(users), len(categories)
    user_category = np.zeros((n_users, n_categories), dtype=np.float32)
    temp_train = train.copy()
    temp_train["user_idx"] = temp_train["user_id"].map(user_to_idx)
    temp_train["cat_idx"] = temp_train["category_name"].astype(str).map(category_to_idx)
    category_counts = temp_train.groupby(["user_idx", "cat_idx"]).size()
    for (user_idx, cat_idx), count in category_counts.items():
        user_category[int(user_idx), int(cat_idx)] = count
    user_category = (user_category + 0.5) / (user_category.sum(axis=1, keepdims=True) + 0.5 * n_categories)
    user_category_norm = user_category / (user_category.max(axis=1, keepdims=True) + 1e-12)

    train2 = train.copy()
    train2["local_time"] = train2["utc_time"] + pd.to_timedelta(train2["timezone_offset"], unit="m")
    train2["hbin"] = (train2["local_time"].dt.hour // 4).astype(int)
    temporal = np.ones((6, n_categories), dtype=np.float32) * 0.5
    train2["cat_idx"] = train2["category_name"].astype(str).map(category_to_idx)
    temporal_counts = train2.groupby(["hbin", "cat_idx"]).size()
    for (hour_bin, cat_idx), count in temporal_counts.items():
        temporal[int(hour_bin), int(cat_idx)] += count
    temporal = temporal / (temporal.sum(axis=0, keepdims=True) + 1e-12)
    temporal_norm = temporal / (temporal.max(axis=0, keepdims=True) + 1e-12)

    train_transition = train.sort_values(["user_id", "utc_time", "venue_id"], kind="stable").reset_index(drop=True).copy()
    train_transition["prev_category"] = train_transition.groupby("user_id")["category_name"].shift(1)
    transition_rows = train_transition.dropna(subset=["prev_category"]).copy()
    transition_rows["prev_cat_idx"] = transition_rows["prev_category"].astype(str).map(category_to_idx)
    transition_rows["current_cat_idx"] = transition_rows["category_name"].astype(str).map(category_to_idx)
    transition_rows = transition_rows.dropna(subset=["prev_cat_idx", "current_cat_idx"])
    transition_matrix = np.ones((n_categories, n_categories), dtype=np.float32) * 0.5
    transition_counts = transition_rows.groupby(["prev_cat_idx", "current_cat_idx"]).size()
    for (prev_idx, current_idx), count in transition_counts.items():
        transition_matrix[int(prev_idx), int(current_idx)] += count
    transition_matrix = transition_matrix / (transition_matrix.sum(axis=1, keepdims=True) + 1e-12)
    transition_norm = transition_matrix / (transition_matrix.max(axis=1, keepdims=True) + 1e-12)

    train_steps = df[(df["split"] == "train") & df["prev_lat"].notna() & df["prev_lon"].notna()].copy()
    lat1 = np.deg2rad(train_steps["prev_lat"].to_numpy())
    lon1 = np.deg2rad(train_steps["prev_lon"].to_numpy())
    lat2 = np.deg2rad(train_steps["latitude"].to_numpy())
    lon2 = np.deg2rad(train_steps["longitude"].to_numpy())
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    distance_km = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    tau = float(np.median(distance_km))

    v_lat = venue_meta["latitude"].to_numpy()
    v_lon = venue_meta["longitude"].to_numpy()
    v_cat = venue_meta["cat_idx"].to_numpy()
    v_pop = venue_meta["popularity_feature"].to_numpy(dtype=np.float32)
    coords = np.deg2rad(venue_meta[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")

    validation_eligible = df[(df["split"] == "validation") & df["venue_id"].isin(train_venues) & df["prev_lat"].notna() & df["prev_local_time"].notna()].copy()
    test_eligible = df[(df["split"] == "test") & df["venue_id"].isin(train_venues) & df["prev_lat"].notna() & df["prev_local_time"].notna()].copy().reset_index(drop=True)

    return {
        "df": df, "train": train, "validation": validation, "test": test,
        "venue_meta": venue_meta, "venue_to_idx": venue_to_idx,
        "categories": categories, "category_to_idx": category_to_idx,
        "users": users, "user_to_idx": user_to_idx,
        "user_category_norm": user_category_norm,
        "temporal_norm": temporal_norm, "transition_norm": transition_norm,
        "tau": tau, "v_lat": v_lat, "v_lon": v_lon, "v_cat": v_cat, "v_pop": v_pop,
        "tree": tree, "validation_eligible": validation_eligible,
        "test_eligible": test_eligible, "n_categories": n_categories,
        "train_steps_n": len(train_steps), "transition_rows_n": len(transition_rows),
    }


def event_features_batch(events: pd.DataFrame, candidate_indices: np.ndarray, state: dict) -> np.ndarray:
    """Vectorized equivalent of the submitted event_features function."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
    v_cat, v_lat, v_lon, v_pop = state["v_cat"], state["v_lat"], state["v_lon"], state["v_pop"]
    cat_indices = v_cat[candidate_indices]
    user_idx = events["user_id"].map(state["user_to_idx"]).to_numpy(dtype=int)
    preference = state["user_category_norm"][user_idx[:, None], cat_indices]

    lat1 = np.deg2rad(events["prev_lat"].to_numpy(dtype=float))[:, None]
    lon1 = np.deg2rad(events["prev_lon"].to_numpy(dtype=float))[:, None]
    lat2 = np.deg2rad(v_lat[candidate_indices])
    lon2 = np.deg2rad(v_lon[candidate_indices])
    a = (np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2)
    distance = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    proximity = np.exp(-distance / state["tau"])
    popularity = v_pop[candidate_indices]

    hour_bin = (events["prev_local_time"].dt.hour.to_numpy(dtype=int) // 4)
    temporal_fit = state["temporal_norm"][hour_bin[:, None], cat_indices]

    previous_category = events["prev_category"].astype(str).map(state["category_to_idx"])
    prev_cat_idx = previous_category.fillna(-1).to_numpy(dtype=int)
    transition_fit = np.zeros(candidate_indices.shape, dtype=np.float32)
    valid = prev_cat_idx >= 0
    if np.any(valid):
        transition_fit[valid] = state["transition_norm"][prev_cat_idx[valid, None], cat_indices[valid]]

    return np.stack([preference, proximity, popularity, temporal_fit, transition_fit], axis=2).astype(np.float32)


def learn_weights(state: dict) -> tuple[np.ndarray, dict]:
    validation_eligible = state["validation_eligible"]
    venue_meta = state["venue_meta"]
    tree = state["tree"]
    v_cat = state["v_cat"]
    venue_to_idx = state["venue_to_idx"]
    rng = np.random.default_rng(RANDOM_SEED)

    pop_prob = np.sqrt(venue_meta["pop_count"].to_numpy(dtype=float) + 1)
    pop_prob = pop_prob / pop_prob.sum()
    cat_to_venues = {ci: venue_meta.index[venue_meta["cat_idx"] == ci].to_numpy() for ci in range(state["n_categories"])}
    val_sample = validation_eligible.sample(n=min(5000, len(validation_eligible)), random_state=42).reset_index(drop=True)
    query_coords = np.deg2rad(val_sample[["prev_lat", "prev_lon"]].to_numpy())
    _, near100 = tree.query(query_coords, k=100)
    val_candidates = np.empty((len(val_sample), 21), dtype=np.int32)

    for j, row in val_sample.iterrows():
        target = venue_to_idx[row["venue_id"]]
        target_cat = v_cat[target]
        negatives = []
        same_category = cat_to_venues[target_cat]
        same_category = same_category[same_category != target]
        if len(same_category) > 0:
            chosen = rng.choice(same_category, size=min(10, len(same_category)), replace=False)
            negatives.extend([int(x) for x in chosen])
        for x in near100[j]:
            x = int(x)
            if x != target and x not in negatives:
                negatives.append(x)
            if len(negatives) >= 15:
                break
        while len(negatives) < 20:
            x = int(rng.choice(len(venue_meta), p=pop_prob))
            if x != target and x not in negatives:
                negatives.append(x)
        val_candidates[j, 0] = target
        val_candidates[j, 1:] = np.asarray(negatives[:20], dtype=np.int32)

    val_features = event_features_batch(val_sample, val_candidates, state)
    diffs = (val_features[:, 0:1, :] - val_features[:, 1:, :]).reshape(-1, 5)

    def objective(w):
        margins = diffs.dot(w)
        pairwise_loss = np.logaddexp(0, -margins).mean()
        return pairwise_loss + ALPHA * np.sum(w ** 2)

    optimization = minimize(objective, x0=np.ones(5) * 0.5,
                            bounds=[(0, None)] * 5, method="L-BFGS-B")
    weights = optimization.x
    weights = weights / weights.sum()
    info = {"success": bool(optimization.success), "message": str(optimization.message),
            "iterations": int(optimization.nit), "objective": float(optimization.fun),
            "alpha_l2": ALPHA, "validation_events_sampled": int(len(val_sample)),
            "pairwise_training_comparisons": int(len(diffs)),
            "features": FEATURE_NAMES, "weights": weights.tolist()}
    if not optimization.success:
        raise RuntimeError(f"Weight optimization failed: {optimization.message}")
    return weights, info


def build_candidate_sets(state: dict) -> np.ndarray:
    test_eligible = state["test_eligible"]
    venue_meta = state["venue_meta"]
    venue_to_idx = state["venue_to_idx"]
    tree = state["tree"]
    top_pop_idx = venue_meta.sort_values(["pop_count", "venue_id"], ascending=[False, True], kind="stable").index[:5000].to_numpy()
    query_coords = np.deg2rad(test_eligible[["prev_lat", "prev_lon"]].to_numpy())
    _, nearest = tree.query(query_coords, k=N_NEAR + 50)
    candidate_sets = np.empty((len(test_eligible), N_CANDIDATES), dtype=np.int32)
    rng = np.random.default_rng(RANDOM_SEED)
    for j, row in test_eligible.iterrows():
        target = venue_to_idx[row["venue_id"]]
        candidates = [target]
        seen = {target}
        for x in nearest[j]:
            x = int(x)
            if x not in seen:
                candidates.append(x)
                seen.add(x)
            if len(candidates) >= 1 + N_NEAR:
                break
        while len(candidates) < N_CANDIDATES:
            sample_popular = rng.choice(top_pop_idx, size=min(400, len(top_pop_idx)), replace=False)
            for x in sample_popular:
                x = int(x)
                if x not in seen:
                    candidates.append(x)
                    seen.add(x)
                if len(candidates) >= N_CANDIDATES:
                    break
        candidate_sets[j] = np.asarray(candidates, dtype=np.int32)
    return candidate_sets


def build_top_candidates(state: dict, candidate_sets: np.ndarray, weights: np.ndarray,
                         chunk_size: int = 256):
    n = len(candidate_sets)
    top_scores = np.empty((n, TOP_M), dtype=np.float64)
    top_features = np.empty((n, TOP_M, 5), dtype=np.float32)
    top_candidate_ids = np.empty((n, TOP_M), dtype=np.int32)
    base_ranks = np.empty(n, dtype=np.int32)
    start = time.perf_counter()
    events = state["test_eligible"]
    for lo in range(0, n, chunk_size):
        hi = min(n, lo + chunk_size)
        ev = events.iloc[lo:hi]
        X = event_features_batch(ev, candidate_sets[lo:hi], state)
        scores = X @ weights
        order = np.argsort(-scores, axis=1, kind="stable")
        base_ranks[lo:hi] = np.argmax(order == 0, axis=1) + 1
        top_idx = order[:, :TOP_M]
        top_candidate_ids[lo:hi] = np.take_along_axis(candidate_sets[lo:hi], top_idx, axis=1)
        top_scores[lo:hi] = np.take_along_axis(scores, top_idx, axis=1)
        top_features[lo:hi] = np.take_along_axis(X, top_idx[:, :, None], axis=1)
    elapsed = time.perf_counter() - start
    return top_scores, top_features, top_candidate_ids, base_ranks, elapsed


def make_evidence(top_features: np.ndarray, weights: np.ndarray,
                  rho: float = RHO, anchor: str = "hybrid"):
    contributions = top_features.astype(np.float64) * weights[None, None, :]
    n, m, f = contributions.shape
    if anchor == "hybrid":
        delta = np.empty_like(contributions)
        delta[:, 0, :] = np.maximum(0.0, contributions[:, 0, :] - contributions[:, 1, :])
        delta[:, 1:, :] = np.maximum(0.0, contributions[:, 1:, :] - contributions[:, 0:1, :])
    elif anchor == "runner_up_common":
        # Every top-M candidate, including the incumbent, is compared to the
        # same fixed runner-up anchor. This is a sensitivity analysis only.
        delta = np.maximum(0.0, contributions - contributions[:, 1:2, :])
    else:
        raise ValueError(f"Unknown anchor: {anchor}")

    totals = delta.sum(axis=2)
    feature_order = np.argsort(-delta, axis=2, kind="stable")
    sorted_delta = np.take_along_axis(delta, feature_order, axis=2)
    cumulative = np.cumsum(sorted_delta, axis=2)
    K = np.full((n, m), f, dtype=np.int8)
    positive = totals > 1e-12
    ratios = np.zeros_like(cumulative)
    ratios[positive] = cumulative[positive] / totals[positive, None]
    first = np.argmax(ratios >= rho, axis=2) + 1
    K[positive] = first[positive]
    return {"contributions": contributions, "delta": delta, "totals": totals,
            "feature_order": feature_order, "sorted_delta": sorted_delta,
            "cumulative": cumulative, "K": K, "anchor": anchor}


def evaluate_policy_full(top_scores: np.ndarray, top_candidate_ids: np.ndarray,
                         target_candidate_id: np.ndarray, base_ranks: np.ndarray,
                         evidence: dict, B: int, lam: float, epsilon: float,
                         eta: float = ETA, identity_control: bool = False):
    n, m = top_scores.shape
    K, totals = evidence["K"], evidence["totals"]
    if identity_control:
        chosen = np.zeros(n, dtype=int)
    else:
        eligible = top_scores >= (top_scores[:, [0]] - epsilon)
        burden = np.maximum(0, K - B)
        adjusted = top_scores - lam * burden - eta * K / evidence["contributions"].shape[2]
        adjusted = np.where(eligible, adjusted, -np.inf)
        chosen = np.argmax(adjusted, axis=1)

    rows = np.arange(n)
    selected_candidate_id = top_candidate_ids[rows, chosen]
    target_selected = selected_candidate_id == target_candidate_id
    new_ranks = base_ranks.copy()
    new_ranks[target_selected] = 1
    shift = (~target_selected) & ((chosen + 1) > base_ranks)
    new_ranks[shift] += 1

    selected_K = K[rows, chosen]
    selected_D = totals[rows, chosen]
    displayed_L = np.where(selected_D > 1e-12, np.minimum(selected_K, B), 0)
    selected_sorted = evidence["sorted_delta"][rows, chosen, :]
    selected_cumulative = np.cumsum(selected_sorted, axis=1)
    displayed_num = np.zeros(n, dtype=float)
    has_display = displayed_L > 0
    displayed_num[has_display] = selected_cumulative[rows[has_display], displayed_L[has_display] - 1]
    F_model = np.divide(displayed_num, selected_D, out=np.zeros(n, dtype=float), where=selected_D > 1e-12)

    base_score = top_scores[:, 0]
    chosen_score = top_scores[rows, chosen]
    score_loss = np.maximum(0.0, base_score - chosen_score)
    selected_over = selected_K > B
    positive_D = selected_D > 1e-12
    promoted = chosen != 0
    promotion_depth = chosen + 1

    metrics = metric_dict(new_ranks)
    result = {
        "B": int(B), "lambda": float(lam), "epsilon": float(epsilon),
        "identity_control": bool(identity_control),
        "metrics": metrics,
        "NDCG_retained_pct": 100.0 * metrics["NDCG@5"] / metric_dict(base_ranks)["NDCG@5"],
        "displayed_units_mean": float(displayed_L.mean()),
        "F_model_mean": float(F_model.mean()),
        "rerank_rate": float(promoted.mean()),
        "budget_violations_after_truncation": int(np.sum(displayed_L > B)),
        "pre_truncation_overbudget_rate_all": float(selected_over.mean()),
        "pre_truncation_overbudget_rate_positive_D": float(np.mean(selected_over & positive_D)),
        "coverage_below_rho_rate_all": float(np.mean(positive_D & (F_model < RHO))),
        "coverage_below_rho_rate_positive_D": float(np.sum(positive_D & (F_model < RHO)) / max(1, np.sum(positive_D))),
        "D_zero_rate": float(np.mean(~positive_D)),
        "score_loss_all": summarize_distribution(score_loss),
        "score_loss_promoted": summarize_distribution(score_loss[promoted]),
        "promotion_depth_promoted": summarize_distribution(promotion_depth[promoted]),
        "promotion_depth_counts": {str(k): int(np.sum(promotion_depth[promoted] == k)) for k in range(1, TOP_M + 1)},
        "score_loss_exceeds_epsilon_count": int(np.sum(score_loss > epsilon + 1e-10)),
        "ranks": new_ranks,
        "chosen_positions": chosen,
        "selected_candidate_ids": selected_candidate_id,
        "selected_D": selected_D,
        "selected_K": selected_K,
        "displayed_L": displayed_L,
        "F_model_per_event": F_model,
        "score_loss_per_event": score_loss,
        "promotion_depth_per_event": promotion_depth,
    }
    return result


def evaluate_budget_matched_posthoc(top_scores, top_candidate_ids, target_candidate_id,
                                    base_ranks, evidence, B: int):
    n = len(base_ranks)
    rows = np.arange(n)
    K0 = evidence["K"][:, 0]
    D0 = evidence["totals"][:, 0]
    L = np.where(D0 > 1e-12, np.minimum(K0, B), 0)
    sorted0 = evidence["sorted_delta"][:, 0, :]
    cumulative0 = np.cumsum(sorted0, axis=1)
    numerator = np.zeros(n, dtype=float)
    has = L > 0
    numerator[has] = cumulative0[rows[has], L[has] - 1]
    F = np.divide(numerator, D0, out=np.zeros(n, dtype=float), where=D0 > 1e-12)
    return {
        "B": int(B), "metrics": metric_dict(base_ranks),
        "NDCG_retained_pct": 100.0,
        "displayed_units_mean": float(L.mean()), "F_model_mean": float(F.mean()),
        "rerank_rate": 0.0, "budget_violations_after_truncation": int(np.sum(L > B)),
        "coverage_below_rho_rate_all": float(np.mean((D0 > 1e-12) & (F < RHO))),
        "coverage_below_rho_rate_positive_D": float(np.sum((D0 > 1e-12) & (F < RHO)) / max(1, np.sum(D0 > 1e-12))),
        "D_zero_rate": float(np.mean(D0 <= 1e-12)),
        "displayed_L": L, "F_model_per_event": F,
    }


def user_level_statistics(user_ids: np.ndarray, base_ranks: np.ndarray,
                          policy_ranks: dict[str, np.ndarray]) -> dict:
    frame = pd.DataFrame({"user_id": user_ids, "RankOnly": ndcg_contribution(base_ranks)})
    for name, ranks in policy_ranks.items():
        frame[name] = ndcg_contribution(ranks)
    user_means = frame.groupby("user_id").mean(numeric_only=True)
    out = {"number_of_users": int(len(user_means)),
           "event_weighted_baseline_NDCG": float(frame["RankOnly"].mean()),
           "user_weighted_baseline_NDCG": float(user_means["RankOnly"].mean()),
           "policies": {}}
    for name in policy_ranks:
        x = user_means["RankOnly"].to_numpy()
        y = user_means[name].to_numpy()
        diff_mean, ci = paired_mean_ci(x, y)
        nz = y - x
        nz = nz[nz != 0]
        ttest = stats.ttest_rel(x, y)
        wtest = stats.wilcoxon(nz) if len(nz) else None
        out["policies"][name] = {
            "user_weighted_NDCG": float(y.mean()),
            "mean_user_level_difference": diff_mean,
            "95pct_CI": ci,
            "paired_t_p": float(ttest.pvalue),
            "wilcoxon_p": float(wtest.pvalue) if wtest is not None else None,
            "nonzero_users": int(len(nz)),
        }
    return out


def load_weight_checkpoint(checkpoint_path: Path, dataset_sha256: str) -> tuple[np.ndarray, dict]:
    """Load the validation-fitted scorer vector distributed with the reproduction package."""
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("The scorer checkpoint was fitted for a different dataset SHA-256")
    if checkpoint.get("features") != FEATURE_NAMES:
        raise ValueError("Feature order in scorer checkpoint does not match FEATURE_NAMES")
    weights = np.asarray(checkpoint.get("weights"), dtype=float)
    if weights.shape != (len(FEATURE_NAMES),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid scorer weights in checkpoint")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("Scorer weights in checkpoint must sum to one")
    training_info = dict(checkpoint.get("training_protocol", {}))
    training_info.update({"success": True, "features": FEATURE_NAMES, "weights": weights.tolist(),
                          "weight_source": "stored validation-trained checkpoint"})
    return weights, training_info


def run_foursquare(dataset_path: Path, weights_path: Path | None = None,
                   refit_weights: bool = False) -> dict:
    print("[Foursquare] Loading and preparing data...", flush=True)
    state = load_and_prepare(dataset_path)
    print({"checkins": len(state["df"]), "users": state["df"]["user_id"].nunique(),
           "venues": state["df"]["venue_id"].nunique(), "train_POIs": len(state["venue_meta"]),
           "eligible_validation": len(state["validation_eligible"]),
           "eligible_test": len(state["test_eligible"]), "tau_km": state["tau"]}, flush=True)

    if refit_weights:
        print("[Foursquare] Re-fitting scorer weights on validation data...", flush=True)
        weights, training_info = learn_weights(state)
        training_info["weight_source"] = "re-fitted from validation sample"
    else:
        checkpoint_path = weights_path or Path(__file__).with_name("foursquare_model_weights.json")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Scorer weight checkpoint not found: {checkpoint_path}")
        print("[Foursquare] Loading validation-trained scorer checkpoint...", flush=True)
        weights, training_info = load_weight_checkpoint(checkpoint_path, sha256_file(dataset_path))
    print("Weights:", dict(zip(FEATURE_NAMES, weights.tolist())), flush=True)

    print("[Foursquare] Building deterministic 500-candidate sets...", flush=True)
    t0 = time.perf_counter()
    candidate_sets = build_candidate_sets(state)
    candidate_build_seconds = time.perf_counter() - t0
    print("Candidate matrix:", candidate_sets.shape, "time:", candidate_build_seconds, flush=True)

    print("[Foursquare] Vectorized test feature scoring and top-40 extraction...", flush=True)
    top_scores, top_features, top_candidate_ids, base_ranks, scoring_seconds = build_top_candidates(
        state, candidate_sets, weights)
    target_candidate_id = candidate_sets[:, 0]
    assert candidate_sets.shape == (len(state["test_eligible"]), N_CANDIDATES)
    print("Scoring completed in", scoring_seconds, "seconds", flush=True)

    print("[Foursquare] Computing evidence and revision experiments...", flush=True)
    t_rerank = time.perf_counter()
    evidence_hybrid = make_evidence(top_features, weights, RHO, anchor="hybrid")
    base_metrics = metric_dict(base_ranks)

    # Budget-matched post-hoc baselines for all three budgets.
    posthoc = {B: evaluate_budget_matched_posthoc(top_scores, top_candidate_ids,
                                                   target_candidate_id, base_ranks,
                                                   evidence_hybrid, B)
               for B in (1, 2, 3)}

    # Original operating policies, retained for comparability with the submitted paper.
    p1 = evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                              base_ranks, evidence_hybrid, B=1, lam=0.06, epsilon=0.08)
    p2 = evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                              base_ranks, evidence_hybrid, B=2, lam=0.03, epsilon=0.04)
    p3 = evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                              base_ranks, evidence_hybrid, B=3, lam=0.0, epsilon=0.0,
                              identity_control=True)

    # Fixed-parameter budget ablation: B varies, lambda and epsilon remain fixed.
    fixed_param_budget = {
        B: evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                                base_ranks, evidence_hybrid, B=B, lam=0.03, epsilon=0.04)
        for B in (1, 2, 3)
    }

    # Component ablation: re-ranking is retained, but the over-budget lambda penalty is removed.
    no_overbudget_penalty = {
        B: evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                                base_ranks, evidence_hybrid, B=B, lam=0.0, epsilon=0.04)
        for B in (1, 2, 3)
    }

    # Alternative common comparison reference: all candidates compared to the runner-up.
    evidence_common = make_evidence(top_features, weights, RHO, anchor="runner_up_common")
    common_anchor_p2 = evaluate_policy_full(top_scores, top_candidate_ids, target_candidate_id,
                                            base_ranks, evidence_common,
                                            B=2, lam=0.03, epsilon=0.04)

    rerank_seconds = time.perf_counter() - t_rerank

    # Find a real promoted case with at least one positive and one negative feature difference.
    promoted_rows = np.where(p1["chosen_positions"] > 0)[0]
    example = None
    contributions = evidence_hybrid["contributions"]
    for idx in promoted_rows:
        chosen_pos = int(p1["chosen_positions"][idx])
        signed = contributions[idx, chosen_pos] - contributions[idx, 0]
        if np.any(signed > 1e-10) and np.any(signed < -1e-10):
            example = {
                "event_index_zero_based": int(idx),
                "promoted_depth_one_based": int(chosen_pos + 1),
                "base_score": float(top_scores[idx, 0]),
                "promoted_score": float(top_scores[idx, chosen_pos]),
                "base_score_loss": float(p1["score_loss_per_event"][idx]),
                "base_rank_of_target_before_promotion": int(base_ranks[idx]),
                "required_positive_evidence_units_K": int(p1["selected_K"][idx]),
                "displayed_positive_evidence_units_L": int(p1["displayed_L"][idx]),
                "displayed_F_model": float(p1["F_model_per_event"][idx]),
                "signed_feature_contribution_differences": {
                    FEATURE_NAMES[j]: float(signed[j]) for j in range(5)
                },
                "positive_only_evidence_differences": {
                    FEATURE_NAMES[j]: float(max(0.0, signed[j])) for j in range(5)
                },
                "note": "One anonymized observed test transition; values are model contribution differences, not causal effects.",
            }
            break

    # Operational timing: report separate scorer and re-ranking stages.
    n_events = len(base_ranks)
    timing = {
        "candidate_generation_seconds": candidate_build_seconds,
        "feature_scoring_and_topM_seconds": scoring_seconds,
        "evidence_and_policy_evaluation_seconds": rerank_seconds,
        "feature_scoring_ms_per_event": 1000.0 * scoring_seconds / n_events,
        "evidence_and_policy_ms_per_event": 1000.0 * rerank_seconds / n_events,
        "evidence_and_policy_over_feature_scoring_pct": 100.0 * rerank_seconds / max(scoring_seconds, 1e-12),
        "note": "Wall-clock measurements from this execution environment; rerun on target hardware for deployment claims.",
    }

    policy_ranks = {"P1": p1["ranks"], "P2": p2["ranks"], "P3": p3["ranks"]}
    user_stats = user_level_statistics(state["test_eligible"]["user_id"].to_numpy(), base_ranks, policy_ranks)

    def compact_policy(p):
        return {k: v for k, v in p.items() if not k.endswith("_per_event") and k not in ("ranks", "chosen_positions", "selected_candidate_ids", "selected_D", "selected_K", "displayed_L", "F_model_per_event", "score_loss_per_event", "promotion_depth_per_event")}

    # Detailed CSV tables, one row per method/policy.
    table_main = []
    table_main.append({"Method": "RankOnly", **base_metrics, "displayed_units_mean": None,
                       "F_model_mean": None, "rerank_rate": 0.0, "NDCG_retained_pct": 100.0})
    for B in (1, 2, 3):
        pp = posthoc[B]
        table_main.append({"Method": f"Budget-matched post-hoc (B={B})",
                           **pp["metrics"], "displayed_units_mean": pp["displayed_units_mean"],
                           "F_model_mean": pp["F_model_mean"], "rerank_rate": 0.0,
                           "NDCG_retained_pct": pp["NDCG_retained_pct"]})
    for label, p in [("CL-MEE P1", p1), ("CL-MEE P2", p2), ("P3 identity control", p3)]:
        table_main.append({"Method": label, **p["metrics"],
                           "displayed_units_mean": p["displayed_units_mean"],
                           "F_model_mean": p["F_model_mean"], "rerank_rate": p["rerank_rate"],
                           "NDCG_retained_pct": p["NDCG_retained_pct"]})

    ablation_rows = []
    for B in (1, 2, 3):
        ablation_rows.append({"Experiment": "Truncation-only budget-matched post-hoc", "B": B,
                              "lambda": None, "epsilon": None,
                              "NDCG@5": posthoc[B]["metrics"]["NDCG@5"],
                              "NDCG_retained_pct": posthoc[B]["NDCG_retained_pct"],
                              "units": posthoc[B]["displayed_units_mean"],
                              "F_model": posthoc[B]["F_model_mean"],
                              "rerank_rate": 0.0})
        for exp_name, exp in [("Fixed lambda/epsilon budget ablation", fixed_param_budget),
                              ("Re-ranking without lambda penalty", no_overbudget_penalty)]:
            p = exp[B]
            ablation_rows.append({"Experiment": exp_name, "B": B,
                                  "lambda": p["lambda"], "epsilon": p["epsilon"],
                                  "NDCG@5": p["metrics"]["NDCG@5"],
                                  "NDCG_retained_pct": p["NDCG_retained_pct"],
                                  "units": p["displayed_units_mean"],
                                  "F_model": p["F_model_mean"],
                                  "rerank_rate": p["rerank_rate"]})

    # Pairwise-reference sensitivity summary.
    anchor_sensitivity = {
        "original_hybrid_anchor_P2": compact_policy(p2),
        "common_runner_up_anchor_P2": compact_policy(common_anchor_p2),
        "interpretation": "The common-runner-up policy is a sensitivity analysis; the primary method remains the hybrid reference convention stated in the paper.",
    }

    # A concise operational summary for original operating points.
    operational = {"P1": compact_policy(p1), "P2": compact_policy(p2), "P3": compact_policy(p3)}

    result = {
        "protocol": {
            "dataset_sha256": sha256_file(dataset_path),
            "checkins": int(len(state["df"])),
            "users": int(state["df"]["user_id"].nunique()),
            "venues_all": int(state["df"]["venue_id"].nunique()),
            "train_POIs": int(len(state["venue_meta"])),
            "eligible_validation_transitions": int(len(state["validation_eligible"])),
            "eligible_test_transitions": int(n_events),
            "candidate_count": int(candidate_sets.shape[1]),
            "top_M": TOP_M,
            "rho": RHO,
            "eta": ETA,
            "negative_sampling": {"same_category": 10, "nearby_to_15_total": 15, "popularity_filled_to": 20,
                                  "test_candidates": "1 target + 249 nearest + 250 popularity-focused"},
            "weight_training": training_info,
            "tau_km": state["tau"],
            "split": "per-user chronological 80/10/10; validation and test targets restricted to training-seen POIs",
            "test_candidate_order": "target first, then nearest candidates, then popularity-focused candidates; stable descending score sort; ties preserve candidate order",
        },
        "base_metrics_event_weighted": base_metrics,
        "main_comparison_table": table_main,
        "budget_matched_posthoc": {str(B): compact_policy(posthoc[B]) for B in (1, 2, 3)},
        "original_policies": {"P1": compact_policy(p1), "P2": compact_policy(p2), "P3": compact_policy(p3)},
        "fixed_parameter_budget_ablation": {str(B): compact_policy(fixed_param_budget[B]) for B in (1, 2, 3)},
        "no_overbudget_penalty_ablation": {str(B): compact_policy(no_overbudget_penalty[B]) for B in (1, 2, 3)},
        "ablation_table": ablation_rows,
        "comparison_anchor_sensitivity": anchor_sensitivity,
        "operational_diagnostics": operational,
        "runtime": timing,
        "user_level_statistics": user_stats,
        "positive_negative_example": example,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset_TSMC2014_NYC.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_outputs"))
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--weights-file", type=Path, default=None,
                        help="Optional JSON file containing validation-trained scorer weights")
    parser.add_argument("--refit-weights", action="store_true",
                        help="Re-run the validation optimizer instead of loading the supplied checkpoint")
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("Environment:", {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "scikit_learn": sklearn.__version__}, flush=True)
    print("Dataset SHA256:", sha256_file(dataset_path), flush=True)

    if not args.skip_simulation:
        print("[Simulation] Running 10 seeds x 800 users x 3 sessions...", flush=True)
        simulation = run_simulation()
        pd.DataFrame(simulation["table"]).to_csv(output_dir / "simulation_comparison.csv", index=False)
    else:
        simulation = None
        for stale_name in ("simulation_comparison.csv", "sensitivity_analysis_summary.csv",
                           "sensitivity_analysis_seed_level.csv", "sensitivity_analysis_protocol.json"):
            stale_path = output_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    foursquare = run_foursquare(dataset_path, weights_path=args.weights_file, refit_weights=args.refit_weights)
    environment = {"python": platform.python_version(),
                   "python_implementation": platform.python_implementation(),
                   "platform": platform.platform(), "system": platform.system(),
                   "release": platform.release(), "machine": platform.machine(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    output = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment,
        "simulation": simulation,
        "foursquare": foursquare,
    }
    output = safe_float(output)
    (output_dir / "revision_results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if simulation is not None:
        sensitivity = simulation["sensitivity_analysis"]
        pd.DataFrame(sensitivity["summary"]).to_csv(output_dir / "sensitivity_analysis_summary.csv", index=False)
        pd.DataFrame(sensitivity["seed_level"]).to_csv(output_dir / "sensitivity_analysis_seed_level.csv", index=False)
        (output_dir / "sensitivity_analysis_protocol.json").write_text(
            json.dumps(safe_float(sensitivity["protocol"] | {"ranges_across_setting_means": sensitivity["ranges_across_setting_means"]}),
                       indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(foursquare["main_comparison_table"]).to_csv(output_dir / "foursquare_main_comparison.csv", index=False)
    pd.DataFrame(foursquare["ablation_table"]).to_csv(output_dir / "foursquare_ablations.csv", index=False)
    anchor_rows = []
    for label, key in (("hybrid", "original_hybrid_anchor_P2"),
                       ("common_runner_up", "common_runner_up_anchor_P2")):
        policy = foursquare["comparison_anchor_sensitivity"][key]
        metrics = policy.get("metrics", {})
        anchor_rows.append({
            "anchor": label,
            "NDCG@5": metrics.get("NDCG@5"),
            "Recall@5": metrics.get("Recall@5"),
            "MRR": metrics.get("MRR"),
            "NDCG_retained_pct": policy.get("NDCG_retained_pct"),
            "displayed_units_mean": policy.get("displayed_units_mean"),
            "F_model_mean": policy.get("F_model_mean"),
            "rerank_rate": policy.get("rerank_rate"),
            "coverage_below_rho_rate_positive_D": policy.get("coverage_below_rho_rate_positive_D"),
            "pre_truncation_overbudget_rate_positive_D": policy.get("pre_truncation_overbudget_rate_positive_D"),
            "budget_violations_after_truncation": policy.get("budget_violations_after_truncation"),
        })
    pd.DataFrame(anchor_rows).to_csv(output_dir / "comparison_anchor_sensitivity.csv", index=False)
    operational_rows = []
    for policy_name in ("P1", "P2"):
        diag = foursquare["operational_diagnostics"][policy_name]
        row = {"Policy": policy_name}
        for group_name in ("score_loss_all", "score_loss_promoted", "promotion_depth_promoted"):
            for stat_name, value in diag[group_name].items():
                row[f"{group_name}_{stat_name}"] = value
        for rate_name in ("pre_truncation_overbudget_rate_all", "pre_truncation_overbudget_rate_positive_D",
                          "coverage_below_rho_rate_all", "coverage_below_rho_rate_positive_D",
                          "budget_violations_after_truncation", "score_loss_exceeds_epsilon_count"):
            row[rate_name] = diag[rate_name]
        operational_rows.append(row)
    pd.DataFrame(operational_rows).to_csv(output_dir / "operational_diagnostics.csv", index=False)
    example_path = output_dir / "illustrative_positive_negative_case.csv"
    if foursquare["positive_negative_example"]:
        ex = foursquare["positive_negative_example"]
        pd.DataFrame([{"Feature": k, "Signed_difference": v,
                       "Positive_evidence_only": ex["positive_only_evidence_differences"][k]}
                      for k, v in ex["signed_feature_contribution_differences"].items()]).to_csv(
                          example_path, index=False)
    elif example_path.exists():
        # Avoid accidentally packaging an example left by an earlier run.
        example_path.unlink()
    checkpoint_path = args.weights_file or Path(__file__).with_name("foursquare_model_weights.json")
    output_files = ["revision_results.json", "foursquare_main_comparison.csv",
                    "foursquare_ablations.csv", "comparison_anchor_sensitivity.csv",
                    "operational_diagnostics.csv"]
    if (output_dir / "illustrative_positive_negative_case.csv").exists():
        output_files.append("illustrative_positive_negative_case.csv")
    if not args.skip_simulation:
        output_files.extend(["simulation_comparison.csv", "sensitivity_analysis_summary.csv",
                             "sensitivity_analysis_seed_level.csv", "sensitivity_analysis_protocol.json"])
    missing_outputs = [name for name in output_files if not (output_dir / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"Expected output files were not created: {missing_outputs}")
    manifest = {
        "dataset_path_argument": str(dataset_path.name),
        "dataset_sha256": sha256_file(dataset_path),
        "output_files": output_files,
        "output_file_sha256": {name: sha256_file(output_dir / name) for name in output_files},
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "weight_source": ("re-fitted from validation sample" if args.refit_weights
                          else "stored validation-trained checkpoint"),
        "weight_checkpoint": (None if args.refit_weights else checkpoint_path.name),
        "weight_checkpoint_sha256": (None if args.refit_weights or not checkpoint_path.exists()
                                      else sha256_file(checkpoint_path)),
        "environment": output["environment"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== FOURSQUARE MAIN COMPARISON ===")
    print(pd.DataFrame(foursquare["main_comparison_table"]).to_string(index=False))
    print("\n=== ABLATIONS ===")
    print(pd.DataFrame(foursquare["ablation_table"]).to_string(index=False))
    print("\n=== RUNTIME ===")
    print(json.dumps(foursquare["runtime"], indent=2))
    if simulation is not None:
        print("\n=== SENSITIVITY RANGES (across setting means) ===")
        print(json.dumps(simulation["sensitivity_analysis"]["ranges_across_setting_means"], indent=2))
    print("\nSaved revision outputs to", output_dir)


if __name__ == "__main__":
    main()
