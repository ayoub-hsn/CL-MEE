#!/usr/bin/env python3
"""Verify the CL-MEE reproduction outputs and their correspondence to main.tex."""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader

EXPECTED_DATASET_SHA256 = "2f39ce698e5bff6683c74bec30c436cc022c160b0f3d73c5268b55491b2445f9"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fail(message: str):
    raise SystemExit("FAIL: " + message)

def check_frame(csv_path: Path, rows: list[dict], label: str, atol: float = 1e-10):
    if not csv_path.exists():
        fail(f"missing {csv_path.name}")
    actual = pd.read_csv(csv_path)
    expected = pd.DataFrame(rows)
    if len(actual) != len(expected):
        fail(f"{label}: CSV has {len(actual)} rows; JSON has {len(expected)}")
    if list(actual.columns) != list(expected.columns):
        fail(f"{label}: CSV columns do not match JSON export")
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            a = pd.to_numeric(actual[col], errors="coerce").to_numpy(dtype=float)
            e = pd.to_numeric(expected[col], errors="coerce").to_numpy(dtype=float)
            same_nan = np.isnan(a) & np.isnan(e)
            ok = same_nan | np.isclose(a, e, rtol=0, atol=atol, equal_nan=True)
            if not bool(np.all(ok)):
                idx = int(np.where(~ok)[0][0])
                fail(f"{label}: mismatch in {col}, row {idx}: CSV={a[idx]} JSON={e[idx]}")
        else:
            if actual[col].fillna("").astype(str).tolist() != expected[col].fillna("").astype(str).tolist():
                fail(f"{label}: text mismatch in column {col}")
    print(f"PASS: {label} CSV matches JSON ({len(actual)} rows)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--paper", type=Path, default=None)
    p.add_argument("--reference-dir", type=Path, default=None)
    p.add_argument("--pdf", type=Path, default=None, help="Optional compiled manuscript PDF to check page count")
    p.add_argument("--max-pages", type=int, default=9)
    p.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("foursquare_model_weights.json"))
    args = p.parse_args()
    results_dir = args.results_dir.resolve()
    dataset = args.dataset.resolve()
    if not dataset.exists(): fail(f"dataset not found: {dataset}")
    actual_hash = sha256_file(dataset)
    if actual_hash != EXPECTED_DATASET_SHA256: fail(f"dataset SHA-256 mismatch: {actual_hash}")
    print("PASS: dataset SHA-256", actual_hash)
    if not args.checkpoint.exists(): fail("weight checkpoint not found")
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("dataset_sha256") != actual_hash: fail("checkpoint dataset hash does not match dataset")
    weights = np.asarray(checkpoint.get("weights", []), dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        fail("invalid checkpoint weights")
    print("PASS: validation-trained scorer checkpoint (5 non-negative weights summing to one)")

    required = ["revision_results.json", "run_manifest.json", "foursquare_main_comparison.csv",
                "foursquare_ablations.csv", "operational_diagnostics.csv",
                "illustrative_positive_negative_case.csv"]
    for name in required:
        if not (results_dir / name).exists(): fail(f"missing output file {name}")
    data = json.loads((results_dir / "revision_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != actual_hash: fail("run manifest dataset hash mismatch")
    if manifest.get("weight_source") != "stored validation-trained checkpoint":
        fail("reference verification expects default checkpoint mode; rerun without --refit-weights")
    if manifest.get("weight_checkpoint_sha256") != sha256_file(args.checkpoint):
        fail("manifest checkpoint hash mismatch")
    runner_path = Path(__file__).with_name("run_revision_experiments.py")
    if manifest.get("script_sha256") != sha256_file(runner_path):
        fail("run manifest script hash does not match the current experiment runner")
    print("PASS: run manifest dataset/checkpoint/script hashes and reference mode")

    f = data["foursquare"]
    if f["protocol"]["eligible_test_transitions"] != 20223: fail("unexpected eligible Foursquare test count")
    if f["protocol"]["dataset_sha256"] != actual_hash: fail("JSON protocol dataset hash mismatch")
    if data.get("simulation") is None: fail("simulation results are missing; run without --skip-simulation")
    check_frame(results_dir / "simulation_comparison.csv", data["simulation"]["table"], "simulation")
    check_frame(results_dir / "foursquare_main_comparison.csv", f["main_comparison_table"], "Foursquare main")
    check_frame(results_dir / "foursquare_ablations.csv", f["ablation_table"], "Foursquare ablations")
    operational_rows = []
    for policy_name in ("P1", "P2"):
        diag = f["operational_diagnostics"][policy_name]
        row = {"Policy": policy_name}
        for group_name in ("score_loss_all", "score_loss_promoted", "promotion_depth_promoted"):
            for stat_name, value in diag[group_name].items():
                row[f"{group_name}_{stat_name}"] = value
        for rate_name in ("pre_truncation_overbudget_rate_all", "pre_truncation_overbudget_rate_positive_D",
                          "coverage_below_rho_rate_all", "coverage_below_rho_rate_positive_D",
                          "budget_violations_after_truncation", "score_loss_exceeds_epsilon_count"):
            row[rate_name] = diag[rate_name]
        operational_rows.append(row)
    check_frame(results_dir / "operational_diagnostics.csv", operational_rows, "operational diagnostics")
    # The signed-contribution CSV is derived from the JSON example.
    ex = f["positive_negative_example"]
    expected_example = pd.DataFrame([
        {"Feature": key, "Signed_difference": value,
         "Positive_evidence_only": ex["positive_only_evidence_differences"][key]}
        for key, value in ex["signed_feature_contribution_differences"].items()
    ])
    check_frame(results_dir / "illustrative_positive_negative_case.csv", expected_example.to_dict("records"), "signed-contribution example")

    p1 = f["original_policies"]["P1"]
    p2 = f["original_policies"]["P2"]
    base = f["base_metrics_event_weighted"]["NDCG@5"]
    p1_ret = p1["metrics"]["NDCG@5"] / base * 100
    p2_ret = p2["metrics"]["NDCG@5"] / base * 100
    if not math.isclose(p1_ret, p1["NDCG_retained_pct"], rel_tol=0, abs_tol=1e-9): fail("P1 NDCG retention calculation mismatch")
    if not math.isclose(p2_ret, p2["NDCG_retained_pct"], rel_tol=0, abs_tol=1e-9): fail("P2 NDCG retention calculation mismatch")
    for name, policy in [("P1", p1), ("P2", p2)]:
        if policy["budget_violations_after_truncation"] != 0: fail(f"{name}: post-truncation budget violation")
        if policy["score_loss_exceeds_epsilon_count"] != 0: fail(f"{name}: observed score loss exceeded epsilon")
    print(f"PASS: P1 NDCG@5={p1['metrics']['NDCG@5']:.9f}; retained={p1_ret:.6f}%")
    print(f"PASS: P2 NDCG@5={p2['metrics']['NDCG@5']:.9f}; retained={p2_ret:.6f}%")
    print("PASS: P1/P2 hard display-budget and score-envelope invariants")

    sim_rows = data["simulation"]["table"]
    sim_by_name = {row["Method"]: row for row in sim_rows}
    sim_base = sim_by_name["RankOnly"]["NDCG@5_mean"]
    sim_clmee = sim_by_name["CL-MEE"]["NDCG@5_mean"]
    sim_retained = sim_clmee / sim_base * 100.0
    if not math.isclose(sim_retained, 99.67796159670992, rel_tol=0, abs_tol=1e-8):
        fail(f"unexpected simulation NDCG retention: {sim_retained}")
    if not math.isclose(sim_by_name["CL-MEE"]["Units"], 1.18225, rel_tol=0, abs_tol=1e-10):
        fail("simulation CL-MEE displayed-unit mean differs from reference")
    if not math.isclose(sim_by_name["Budget-matched post-hoc"]["Units"], 1.1759583333333332, rel_tol=0, abs_tol=1e-10):
        fail("simulation budget-matched displayed-unit mean differs from reference")
    print(f"PASS: simulation reference metrics (24,000 sessions; NDCG retained={sim_retained:.6f}%)")

    # Ensure operational summaries include both all-event and promoted-event P1/P2 distributions.
    for name, policy in [("P1", p1), ("P2", p2)]:
        diag = f["operational_diagnostics"][name]
        for group in ("score_loss_all", "score_loss_promoted", "promotion_depth_promoted"):
            if not isinstance(diag.get(group), dict) or "max" not in diag[group]:
                fail(f"{name}: missing operational distribution {group}")
    if f["operational_diagnostics"]["P2"]["score_loss_all"]["max"] <= 0:
        fail("P2 all-event score-loss distribution is missing or invalid")
    print("PASS: P1/P2 all-event and promoted-event operational distributions")

    if args.reference_dir:
        ref_path = args.reference_dir.resolve() / "revision_results.json"
        if not ref_path.exists(): fail("reference revision_results.json not found")
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        # Compare metrics that should be stable with the supplied checkpoint. Runtime and platform metadata are excluded.
        ref_f = ref["foursquare"]
        pairs = [
            ("RankOnly NDCG", f["base_metrics_event_weighted"]["NDCG@5"], ref_f["base_metrics_event_weighted"]["NDCG@5"]),
            ("P1 NDCG", p1["metrics"]["NDCG@5"], ref_f["original_policies"]["P1"]["metrics"]["NDCG@5"]),
            ("P2 NDCG", p2["metrics"]["NDCG@5"], ref_f["original_policies"]["P2"]["metrics"]["NDCG@5"]),
            ("P1 retention", p1_ret, ref_f["original_policies"]["P1"]["NDCG_retained_pct"]),
            ("P2 retention", p2_ret, ref_f["original_policies"]["P2"]["NDCG_retained_pct"]),
        ]
        for label, actual, expected in pairs:
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=5e-7):
                fail(f"{label} differs from reference: actual={actual}, expected={expected}")
        print("PASS: key Foursquare metrics match the packaged reference run")

    if args.paper:
        paper = args.paper.resolve()
        if not paper.exists(): fail(f"paper source not found: {paper}")
        tex = paper.read_text(encoding="utf-8")
        required_fragments = [
            "retains 99.984\\% of baseline NDCG@5",
            "CL-MEE $P_1$ & 1 & .156524 & 1.000 & .912 & 10.99\\%",
            "CL-MEE $P_2$ & 2 & .156732 & 1.270 & .966 & 1.25\\%",
            "$P_1$ retains 99.984\\% of baseline NDCG@5",
            "2 & .156549 & .156732 & .156531 & 1.270 & .966 & 1.25\\%",
            "99.68\\% of baseline NDCG@5",
            "Budget-matched post-hoc & $.1991\\pm.0087$ & $.13103\\pm.00661$",
            "CL-MEE & $.1991\\pm.0087$ & $.13061\\pm.00657$",
            "For $P_1$, all-event loss has mean .00340",
            "promoted-event loss has mean .03089, median .03007",
            "For $P_2$, all-event loss has mean .00015",
            "Pre-truncation $K>B$ occurs in 1.37\\% of all events",
            "comparable as a relative display-burden heuristic",
            "run took 8.76 s for candidate generation",
            "0.76 s for evidence/policy evaluation",
            "0.0377 ms per event",
            "\\keywords{Explainable recommender systems, smart tourism, POI recommendation}",
            "The remainder of the paper is organized as follows.",
            "Future work includes user studies",
        ]
        for fragment in required_fragments:
            if fragment not in tex: fail("manuscript metric not synchronized: " + fragment)
        if "99.995\\%" in tex or ".156542" in tex:
            fail("stale P1 metric remains in main.tex")
        runtime_ref_path = Path(__file__).with_name("runtime_reference_windows.json")
        if not runtime_ref_path.exists(): fail("runtime_reference_windows.json is missing")
        runtime_ref = json.loads(runtime_ref_path.read_text(encoding="utf-8"))["runtime"]
        expected_runtime = [
            ("candidate_generation_seconds", 8.762423899956048),
            ("feature_scoring_and_topM_seconds", 1.9017080000485294),
            ("evidence_and_policy_evaluation_seconds", 0.7623891000403091),
        ]
        for key, expected in expected_runtime:
            if not math.isclose(runtime_ref[key], expected, rel_tol=0, abs_tol=1e-12):
                fail(f"runtime reference mismatch for {key}")
        print("PASS: main.tex contains corrected P1/P2, simulation, operational, runtime, and reviewer-2 revisions")

    if args.pdf:
        pdf_path = args.pdf.resolve()
        if not pdf_path.exists(): fail(f"PDF not found: {pdf_path}")
        page_count = len(PdfReader(str(pdf_path)).pages)
        if page_count != args.max_pages:
            fail(f"PDF page count is {page_count}; expected exactly {args.max_pages}")
        print(f"PASS: compiled PDF page count ({page_count} pages; limit {args.max_pages})")

    report = [
        "CL-MEE reproduction verification report",
        "=======================================",
        f"Dataset SHA-256: {actual_hash}",
        f"P1 NDCG@5: {p1['metrics']['NDCG@5']:.12f} ({p1_ret:.8f}% retained)",
        f"P2 NDCG@5: {p2['metrics']['NDCG@5']:.12f} ({p2_ret:.8f}% retained)",
        f"Simulation NDCG retention: {sim_retained:.8f}% over 24,000 sessions",
        "P1/P2 all-event and promoted-event operational distributions: present",
        "All JSON/CSV, dataset, checkpoint, invariant, and requested manuscript checks passed.",
    ]
    if args.pdf:
        report.append(f"PDF page count: {page_count} pages (limit {args.max_pages})")
    (results_dir / "verification_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\nALL CHECKS PASSED. Report:", results_dir / "verification_report.txt")

if __name__ == "__main__":
    main()
