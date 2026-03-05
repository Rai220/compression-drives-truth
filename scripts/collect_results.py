#!/usr/bin/env python3
"""
Collect all experiment results into a master CSV and generate Markdown tables for the paper.

Convention: ΔLoss = Loss(incorrect) - Loss(correct)
  ΔLoss > 0 → model prefers correct (truth bias)
  ΔLoss < 0 → model prefers incorrect

Usage:
    python scripts/collect_results.py [--results-dir results] [--output-csv results_master.csv]
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_DIR = Path("results")
OUTPUT_CSV = Path("results_master.csv")
OUTPUT_TABLES = Path("scripts/tables.md")


def parse_run_name(name: str) -> dict:
    """Extract experiment type, proportion, model size, seed from directory name."""
    info = {"run_name": name, "experiment": None, "proportion": None,
            "model_size": None, "seed": None}

    # Condition experiments: condC_50_50_tiny_seed42
    m = re.match(r"cond([A-Z])_(\d+)_(\d+)_(\w+)_seed(\d+)", name)
    if m:
        info["experiment"] = f"cond{m.group(1)}"
        correct_pct = int(m.group(2))
        info["proportion"] = correct_pct / 100
        info["model_size"] = m.group(4)
        info["seed"] = int(m.group(5))
        return info

    # Observed: observed_50_tiny_seed42
    m = re.match(r"observed_(\d+)_(\w+)_seed(\d+)", name)
    if m:
        info["experiment"] = "observed"
        info["proportion"] = int(m.group(1)) / 100  # observation %
        info["model_size"] = m.group(2)
        info["seed"] = int(m.group(3))
        return info

    # Coherent with proportions: coherent_20_80_tiny_seed42
    m = re.match(r"coherent_(\d+)_(\d+)_(\w+)_seed(\d+)", name)
    if m:
        info["experiment"] = "coherent"
        correct_pct = int(m.group(1))
        info["proportion"] = correct_pct / 100
        info["model_size"] = m.group(3)
        info["seed"] = int(m.group(4))
        return info

    # Contradictory: contradictory_50_50_tiny_seed42
    m = re.match(r"contradictory_(\d+)_(\d+)_(\w+)_seed(\d+)", name)
    if m:
        info["experiment"] = "contradictory"
        correct_pct = int(m.group(1))
        info["proportion"] = correct_pct / 100
        info["model_size"] = m.group(3)
        info["seed"] = int(m.group(4))
        return info

    # Mixed: mixed_50_50_tiny_seed43 or mixed_50_50_tiny (seed42, legacy)
    m = re.match(r"mixed_(\d+)_(\d+)_(\w+?)(?:_seed(\d+))?$", name)
    if m:
        info["experiment"] = "mixed"
        correct_pct = int(m.group(1))
        info["proportion"] = correct_pct / 100
        info["model_size"] = m.group(3)
        info["seed"] = int(m.group(4)) if m.group(4) else 42
        return info

    # Baseline: baseline_correct_tiny
    m = re.match(r"baseline_correct_(\w+)", name)
    if m:
        info["experiment"] = "baseline"
        info["proportion"] = 1.0
        info["model_size"] = m.group(1)
        info["seed"] = 42
        return info

    return info


def load_eval(run_dir: Path) -> dict | None:
    """Load eval results from eval_perplexity.json or eval_results.json."""
    for fname in ["eval_perplexity.json", "eval_results.json"]:
        p = run_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def load_config(run_dir: Path) -> dict | None:
    p = run_dir / "config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def collect_all(results_dir: Path) -> list[dict]:
    """Collect all results into a flat list of records."""
    records = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = parse_run_name(run_dir.name)
        if info["experiment"] is None:
            continue

        eval_data = load_eval(run_dir)
        config = load_config(run_dir)

        if eval_data is None:
            print(f"WARNING: no eval file in {run_dir.name}, skipping", file=sys.stderr)
            continue

        correct_loss = eval_data["correct_loss"]
        incorrect_loss = eval_data["incorrect_loss"]
        # Unified convention: ΔLoss = Loss(incorrect) - Loss(correct)
        delta_loss = incorrect_loss - correct_loss

        record = {
            "run_name": info["run_name"],
            "experiment": info["experiment"],
            "proportion": info["proportion"],
            "model_size": info["model_size"],
            "seed": info["seed"],
            "correct_loss": correct_loss,
            "incorrect_loss": incorrect_loss,
            "correct_ppl": eval_data.get("correct_ppl"),
            "incorrect_ppl": eval_data.get("incorrect_ppl"),
            "delta_loss": delta_loss,
            "prefers": "correct" if delta_loss > 0 else "incorrect",
            "n_params": config.get("n_params") if config else None,
            "max_steps": config.get("max_steps") if config else None,
        }
        records.append(record)
    return records


def write_csv(records: list[dict], path: Path):
    if not records:
        return
    fields = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {len(records)} records to {path}")


def group_by(records, key_fn):
    """Group records by a key function, preserving order of first appearance."""
    groups = {}
    order = []
    for r in records:
        k = key_fn(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    return [(k, groups[k]) for k in order]


def stats_for_group(recs: list[dict]) -> dict:
    """Compute mean, std, seed counts for a group of records."""
    deltas = [r["delta_loss"] for r in recs]
    cor = [r["correct_loss"] for r in recs]
    inc = [r["incorrect_loss"] for r in recs]
    n = len(deltas)
    seeds_correct = sum(1 for d in deltas if d > 0)

    result = {
        "n_seeds": n,
        "mean_delta": np.mean(deltas),
        "std_delta": np.std(deltas, ddof=1) if n > 1 else 0.0,
        "mean_correct_loss": np.mean(cor),
        "std_correct_loss": np.std(cor, ddof=1) if n > 1 else 0.0,
        "mean_incorrect_loss": np.mean(inc),
        "std_incorrect_loss": np.std(inc, ddof=1) if n > 1 else 0.0,
        "seeds_correct": seeds_correct,
        "seeds_total": n,
    }

    # Binomial test: probability of observing this many seeds preferring correct by chance
    if n > 0:
        result["binom_p"] = stats.binomtest(seeds_correct, n, 0.5).pvalue
    else:
        result["binom_p"] = 1.0

    # Bootstrap CI for mean delta
    if n >= 4:
        rng = np.random.RandomState(42)
        boot_means = []
        for _ in range(10000):
            sample = rng.choice(deltas, size=n, replace=True)
            boot_means.append(np.mean(sample))
        result["ci_lower"] = np.percentile(boot_means, 2.5)
        result["ci_upper"] = np.percentile(boot_means, 97.5)
    else:
        result["ci_lower"] = None
        result["ci_upper"] = None

    return result


def fmt_pm(mean, std, decimals=4):
    return f"{mean:+.{decimals}f} +/- {std:.{decimals}f}"


def fmt_ci(s):
    if s["ci_lower"] is not None:
        return f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
    return "N/A"


def generate_tables(records: list[dict]) -> str:
    """Generate all Markdown tables for the paper."""
    lines = []

    def heading(title):
        lines.append(f"\n## {title}\n")

    # ===== Table 1: Mixed (random errors) by proportion =====
    heading("Table 1: Truth Bias with Random Errors")
    mixed = [r for r in records if r["experiment"] == "mixed"]
    baseline = [r for r in records if r["experiment"] == "baseline"]

    lines.append("| Proportion (corr/incorr) | Loss (correct) | Loss (incorrect) | DLoss | 95% CI | Seeds -> correct | p (binom) |")
    lines.append("|---|---|---|---|---|---|---|")

    if baseline:
        bs = stats_for_group(baseline)
        lines.append(f"| 100/0 (baseline) | {bs['mean_correct_loss']:.4f} | {bs['mean_incorrect_loss']:.4f} | {bs['mean_delta']:+.4f} | N/A | {bs['seeds_correct']}/{bs['seeds_total']} | - |")

    for prop in [0.5, 0.4, 0.3, 0.2, 0.1]:
        group = [r for r in mixed if abs(r["proportion"] - prop) < 0.01]
        if not group:
            continue
        s = stats_for_group(group)
        pct = int(prop * 100)
        lines.append(
            f"| {pct}/{100-pct} "
            f"| {s['mean_correct_loss']:.4f} +/- {s['std_correct_loss']:.4f} "
            f"| {s['mean_incorrect_loss']:.4f} +/- {s['std_incorrect_loss']:.4f} "
            f"| {fmt_pm(s['mean_delta'], s['std_delta'])} "
            f"| {fmt_ci(s)} "
            f"| {s['seeds_correct']}/{s['seeds_total']} "
            f"| {s['binom_p']:.4f} |"
        )

    # ===== Table 2: Error coherence spectrum (50/50) =====
    heading("Table 2: Error Coherence Spectrum (50/50)")
    lines.append("| Error type | Loss (correct) | Loss (incorrect) | DLoss | 95% CI | Seeds -> correct | p (binom) |")
    lines.append("|---|---|---|---|---|---|---|")

    for exp_name, label in [("mixed", "Random"), ("contradictory", "Contradictory"), ("coherent", "Coherent")]:
        group = [r for r in records if r["experiment"] == exp_name and abs(r["proportion"] - 0.5) < 0.01]
        if not group:
            lines.append(f"| {label} | - | - | - | - | - | - | *(no eval data)* |")
            continue
        s = stats_for_group(group)
        lines.append(
            f"| {label} "
            f"| {s['mean_correct_loss']:.4f} +/- {s['std_correct_loss']:.4f} "
            f"| {s['mean_incorrect_loss']:.4f} +/- {s['std_incorrect_loss']:.4f} "
            f"| {fmt_pm(s['mean_delta'], s['std_delta'])} "
            f"| {fmt_ci(s)} "
            f"| {s['seeds_correct']}/{s['seeds_total']} "
            f"| {s['binom_p']:.4f} |"
        )

    # ===== Table 3: Random vs Coherent by proportion =====
    heading("Table 3: Random vs Coherent Errors by Proportion")
    lines.append("| Proportion | Random DLoss | Coherent DLoss | Random -> | Coherent -> |")
    lines.append("|---|---|---|---|---|")

    for prop in [0.5, 0.4, 0.3, 0.2]:
        m_group = [r for r in records if r["experiment"] == "mixed" and abs(r["proportion"] - prop) < 0.01]
        c_group = [r for r in records if r["experiment"] == "coherent" and abs(r["proportion"] - prop) < 0.01]
        ms = stats_for_group(m_group) if m_group else None
        cs = stats_for_group(c_group) if c_group else None
        pct = int(prop * 100)
        m_delta = fmt_pm(ms['mean_delta'], ms['std_delta']) if ms else "N/A"
        c_delta = fmt_pm(cs['mean_delta'], cs['std_delta']) if cs else "N/A"
        m_pref = f"correct ({ms['seeds_correct']}/{ms['seeds_total']})" if ms else "N/A"
        c_pref = f"{'correct' if cs and cs['seeds_correct'] > cs['seeds_total']/2 else 'incorrect'} ({cs['seeds_correct']}/{cs['seeds_total']})" if cs else "N/A"
        lines.append(f"| {pct}/{100-pct} | {m_delta} | {c_delta} | {m_pref} | {c_pref} |")

    # ===== Table 4: Observations (Phase 2) =====
    heading("Table 4: Observations Effect (Coherent Errors, 50/50)")
    obs = [r for r in records if r["experiment"] == "observed"]
    lines.append("| Observation % | Avg DLoss | 95% CI | Seeds -> correct | Avg Loss (correct) | p (binom) |")
    lines.append("|---|---|---|---|---|---|")

    for obs_pct in [0, 10, 25, 50, 100]:
        group = [r for r in obs if abs(r["proportion"] - obs_pct / 100) < 0.01]
        if not group:
            continue
        s = stats_for_group(group)
        lines.append(
            f"| {obs_pct}% "
            f"| {fmt_pm(s['mean_delta'], s['std_delta'])} "
            f"| {fmt_ci(s)} "
            f"| {s['seeds_correct']}/{s['seeds_total']} "
            f"| {s['mean_correct_loss']:.4f} "
            f"| {s['binom_p']:.4f} |"
        )

    # ===== Table 5: Conditions C/D/E (Phase 3) =====
    heading("Table 5: Falsifiability Spectrum (50/50, 4 seeds)")

    # Use observed_0 for condition A, observed_50 for condition B
    cond_a = [r for r in records if r["experiment"] == "observed" and abs(r["proportion"]) < 0.01]
    cond_b = [r for r in records if r["experiment"] == "observed" and abs(r["proportion"] - 0.5) < 0.01]
    cond_c = [r for r in records if r["experiment"] == "condC"]
    cond_d = [r for r in records if r["experiment"] == "condD"]
    cond_e = [r for r in records if r["experiment"] == "condE"]

    lines.append("| Condition | Description | Avg DLoss | 95% CI | Seeds -> correct | p (binom) |")
    lines.append("|---|---|---|---|---|---|")

    for label, desc, group in [
        ("A", "No observations", cond_a),
        ("B", "Bare discrepancies (50% obs)", cond_b),
        ("E", "Vague predictions", cond_e),
        ("C", "Ad hoc escape hatches", cond_c),
        ("D", "Systematic correction", cond_d),
    ]:
        if not group:
            lines.append(f"| {label} | {desc} | N/A | N/A | N/A | N/A |")
            continue
        s = stats_for_group(group)
        lines.append(
            f"| {label} | {desc} "
            f"| {fmt_pm(s['mean_delta'], s['std_delta'])} "
            f"| {fmt_ci(s)} "
            f"| {s['seeds_correct']}/{s['seeds_total']} "
            f"| {s['binom_p']:.4f} |"
        )

    # ===== Summary statistics =====
    heading("Summary: Combined Binomial Test")
    # All mixed seeds 50/50 through 20/80
    mixed_positive = [r for r in records if r["experiment"] == "mixed"
                      and r["proportion"] >= 0.2 and r["proportion"] <= 0.5]
    n_total = len(mixed_positive)
    n_correct = sum(1 for r in mixed_positive if r["delta_loss"] > 0)
    if n_total > 0:
        p_val = stats.binomtest(n_correct, n_total, 0.5).pvalue
        lines.append(f"Mixed 20/80-50/50: {n_correct}/{n_total} seeds prefer correct, p = {p_val:.2e}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-tables", type=Path, default=OUTPUT_TABLES)
    args = parser.parse_args()

    records = collect_all(args.results_dir)
    if not records:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    print(f"Collected {len(records)} experiment runs")

    # Write CSV
    write_csv(records, args.output_csv)

    # Generate tables
    tables = generate_tables(records)
    args.output_tables.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tables, "w") as f:
        f.write(tables)
    print(f"Wrote tables to {args.output_tables}")

    # Print tables to stdout
    print("\n" + "=" * 70)
    print(tables)


if __name__ == "__main__":
    main()
