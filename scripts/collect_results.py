#!/usr/bin/env python3
"""
Collect experiment artifacts into a master CSV and paired-first Markdown tables.

Convention:
  ΔLoss = Loss(incorrect) - Loss(correct)
  ΔLoss > 0 -> model prefers correct
  ΔLoss < 0 -> model prefers incorrect
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_DIR = Path("results")
OUTPUT_CSV = Path("results_master.csv")
OUTPUT_TABLES = Path("scripts/tables.md")


def parse_run_name(name: str) -> dict:
    """Extract experiment metadata from a result directory name."""
    info = {
        "run_name": name,
        "experiment": None,
        "proportion": None,
        "model_size": None,
        "seed": None,
        "variant": None,
    }

    patterns = [
        (r"cond([A-Z])_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": f"cond{m.group(1)}",
            "proportion": int(m.group(2)) / 100,
            "model_size": m.group(4),
            "seed": int(m.group(5)),
        }),
        (r"observed_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "observed",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(2),
            "seed": int(m.group(3)),
        }),
        (r"world_multialt(\d+)_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "world_multialt",
            "variant": int(m.group(1)),
            "proportion": int(m.group(2)) / 100,
            "model_size": m.group(4),
            "seed": int(m.group(5)),
        }),
        (r"world_(random|coherent|contradictory)_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": f"world_{m.group(1)}",
            "proportion": int(m.group(2)) / 100,
            "model_size": m.group(4),
            "seed": int(m.group(5)),
        }),
        (r"crossdomain_(\d+)pct_(\w+)_seed(\d+)", lambda m: {
            "experiment": "crossdomain",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(2),
            "seed": int(m.group(3)),
        }),
        (r"coherent_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "coherent",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(3),
            "seed": int(m.group(4)),
        }),
        (r"multirule_(\d+)_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "multirule",
            "variant": int(m.group(1)),
            "proportion": int(m.group(2)) / 100,
            "model_size": m.group(4),
            "seed": int(m.group(5)),
        }),
        (r"chained_truncated_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "chained_truncated",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(3),
            "seed": int(m.group(4)),
        }),
        (r"chained_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "chained",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(3),
            "seed": int(m.group(4)),
        }),
        (r"contradictory_(\d+)_(\d+)_(\w+)_seed(\d+)", lambda m: {
            "experiment": "contradictory",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(3),
            "seed": int(m.group(4)),
        }),
        (r"mixed_(\d+)_(\d+)_(\w+?)(?:_seed(\d+))?$", lambda m: {
            "experiment": "mixed",
            "proportion": int(m.group(1)) / 100,
            "model_size": m.group(3),
            "seed": int(m.group(4)) if m.group(4) else 42,
        }),
        (r"baseline_correct_(\w+)", lambda m: {
            "experiment": "baseline",
            "proportion": 1.0,
            "model_size": m.group(1),
            "seed": 42,
        }),
    ]

    for pattern, builder in patterns:
        match = re.match(pattern, name)
        if match:
            info.update(builder(match))
            return info

    return info


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config(run_dir: Path) -> dict | None:
    return load_json_if_exists(run_dir / "config.json")


def select_primary_paired_file(info: dict, run_dir: Path) -> tuple[str | None, str | None]:
    """Return (filename, evaluation family) for the main paired artifact."""
    candidates = []
    if info["experiment"] == "multirule":
        candidates = [
            ("eval_paired_matched.json", "matched_multirule"),
            ("eval_paired.json", "legacy_transfer_random"),
        ]
    else:
        candidates = [("eval_paired.json", "standard")]

    for filename, family in candidates:
        if (run_dir / filename).exists():
            return filename, family
    return None, None


def collect_all(results_dir: Path) -> list[dict]:
    """Collect all result directories into flat records."""
    records = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        info = parse_run_name(run_dir.name)
        if info["experiment"] is None:
            continue

        config = load_config(run_dir)
        legacy_corpus = load_json_if_exists(run_dir / "eval_perplexity.json") or load_json_if_exists(run_dir / "eval_results.json")
        deterministic_corpus = load_json_if_exists(run_dir / "eval_perplexity_full.json")
        paired_filename, paired_family = select_primary_paired_file(info, run_dir)
        paired = load_json_if_exists(run_dir / paired_filename) if paired_filename else None

        record = {
            "run_name": info["run_name"],
            "experiment": info["experiment"],
            "proportion": info["proportion"],
            "model_size": info["model_size"],
            "seed": info["seed"],
            "variant": info["variant"],
            "n_params": config.get("n_params") if config else None,
            "max_steps": config.get("max_steps") if config else None,
            "corpus_eval_legacy_file": "eval_perplexity.json" if (run_dir / "eval_perplexity.json").exists() else ("eval_results.json" if (run_dir / "eval_results.json").exists() else None),
            "corpus_eval_full_file": "eval_perplexity_full.json" if deterministic_corpus else None,
            "paired_eval_file": paired_filename,
            "paired_eval_family": paired_family,
            "legacy_correct_loss": None,
            "legacy_incorrect_loss": None,
            "legacy_delta_loss": None,
            "legacy_prefers": None,
            "full_correct_loss": None,
            "full_incorrect_loss": None,
            "full_delta_loss": None,
            "full_prefers": None,
            "pair_accuracy": None,
            "paired_delta": None,
            "paired_sum_accuracy": None,
            "paired_sum_delta": None,
            "paired_length_matched_accuracy": None,
            "paired_length_matched_delta": None,
            "paired_completion_len_correct_mean": None,
            "paired_completion_len_incorrect_mean": None,
            "n_pairs": None,
            "wilcoxon_p": None,
        }

        if legacy_corpus:
            record["legacy_correct_loss"] = legacy_corpus["correct_loss"]
            record["legacy_incorrect_loss"] = legacy_corpus["incorrect_loss"]
            record["legacy_delta_loss"] = legacy_corpus["incorrect_loss"] - legacy_corpus["correct_loss"]
            record["legacy_prefers"] = "correct" if record["legacy_delta_loss"] > 0 else "incorrect"

        if deterministic_corpus:
            full_data = deterministic_corpus.get("example_block_estimate", deterministic_corpus)
            record["full_correct_loss"] = full_data["correct_loss"]
            record["full_incorrect_loss"] = full_data["incorrect_loss"]
            record["full_delta_loss"] = full_data["incorrect_loss"] - full_data["correct_loss"]
            record["full_prefers"] = "correct" if record["full_delta_loss"] > 0 else "incorrect"

        if paired:
            record["pair_accuracy"] = paired.get("pair_accuracy")
            record["paired_delta"] = paired.get("delta")
            record["paired_sum_accuracy"] = paired.get("robustness", {}).get("sum_nll", {}).get("pair_accuracy")
            record["paired_sum_delta"] = paired.get("robustness", {}).get("sum_nll", {}).get("delta")
            record["paired_length_matched_accuracy"] = paired.get("robustness", {}).get("length_matched_mean_nll", {}).get("pair_accuracy")
            record["paired_length_matched_delta"] = paired.get("robustness", {}).get("length_matched_mean_nll", {}).get("delta")
            record["paired_completion_len_correct_mean"] = paired.get("robustness", {}).get("completion_lengths", {}).get("correct_mean")
            record["paired_completion_len_incorrect_mean"] = paired.get("robustness", {}).get("completion_lengths", {}).get("incorrect_mean")
            record["n_pairs"] = paired.get("n_pairs")
            record["wilcoxon_p"] = paired.get("wilcoxon_p")

        records.append(record)

    return records


def write_csv(records: list[dict], path: Path):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {path}")


def filter_records(records: list[dict], **criteria) -> list[dict]:
    return [
        r for r in records
        if all(r.get(key) == value for key, value in criteria.items())
    ]


def filter_prop(records: list[dict], proportion: float) -> list[dict]:
    return [r for r in records if r["proportion"] is not None and abs(r["proportion"] - proportion) < 1e-9]


def summarize_numeric(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def bootstrap_ci(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def summarize_corpus_group(records: list[dict], field_prefix: str) -> dict | None:
    valid = [r for r in records if r[f"{field_prefix}_delta_loss"] is not None]
    if not valid:
        return None
    deltas = [r[f"{field_prefix}_delta_loss"] for r in valid]
    cor = [r[f"{field_prefix}_correct_loss"] for r in valid]
    inc = [r[f"{field_prefix}_incorrect_loss"] for r in valid]
    ci_lo, ci_hi = bootstrap_ci(deltas)
    seeds_correct = sum(1 for d in deltas if d > 0)
    return {
        "n": len(valid),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(valid) > 1 else 0.0,
        "mean_correct": float(np.mean(cor)),
        "std_correct": float(np.std(cor, ddof=1)) if len(valid) > 1 else 0.0,
        "mean_incorrect": float(np.mean(inc)),
        "std_incorrect": float(np.std(inc, ddof=1)) if len(valid) > 1 else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "seeds_correct": seeds_correct,
        "binom_p": stats.binomtest(seeds_correct, len(valid), 0.5).pvalue if valid else None,
    }


def summarize_paired_group(
    records: list[dict],
    accuracy_field: str = "pair_accuracy",
    delta_field: str = "paired_delta",
) -> dict | None:
    valid = [r for r in records if r[accuracy_field] is not None and r[delta_field] is not None]
    if not valid:
        return None
    accs = [r[accuracy_field] for r in valid]
    deltas = [r[delta_field] for r in valid]
    ci_lo, ci_hi = bootstrap_ci(deltas)
    seeds_correct = sum(1 for a in accs if a > 0.5)
    return {
        "n": len(valid),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs, ddof=1)) if len(valid) > 1 else 0.0,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(valid) > 1 else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "min_pairs": min(r["n_pairs"] for r in valid if r["n_pairs"] is not None),
        "max_pairs": max(r["n_pairs"] for r in valid if r["n_pairs"] is not None),
        "seeds_correct": seeds_correct,
        "binom_p": stats.binomtest(seeds_correct, len(valid), 0.5).pvalue if valid else None,
    }


def fmt_pm(mean: float, std: float, decimals: int = 4, pct: bool = False) -> str:
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    return f"{mean * scale:.{decimals}f}{suffix} +/- {std * scale:.{decimals}f}{suffix}"


def fmt_signed(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:+.{decimals}f} +/- {std:.{decimals}f}"


def fmt_ci(ci_lo: float | None, ci_hi: float | None, decimals: int = 4) -> str:
    if ci_lo is None or ci_hi is None:
        return "N/A"
    return f"[{ci_lo:+.{decimals}f}, {ci_hi:+.{decimals}f}]"


def generate_tables(records: list[dict]) -> str:
    lines: list[str] = []

    def heading(title: str):
        lines.append(f"\n## {title}\n")

    tiny = [r for r in records if r["model_size"] == "tiny"]

    heading("Paired Random Proportions")
    lines.append("| Proportion | Pair accuracy | Avg ΔLoss (paired) | Seed directions | n pairs |")
    lines.append("|---|---|---|---|---|")
    for prop in [0.5, 0.4, 0.3, 0.2, 0.1]:
        group = summarize_paired_group(filter_prop(filter_records(tiny, experiment="mixed"), prop))
        if not group:
            continue
        pct = int(prop * 100)
        lines.append(
            f"| {pct}/{100-pct} | {fmt_pm(group['mean_acc'], group['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(group['mean_delta'], group['std_delta'])} "
            f"| {group['seeds_correct']}/{group['n']} > 50% "
            f"| {group['min_pairs']}-{group['max_pairs']} |"
        )

    heading("Corpus-Level Random Proportions (Legacy Window Estimate)")
    lines.append("| Proportion | Loss correct | Loss incorrect | ΔLoss | 95% CI | Seeds -> correct |")
    lines.append("|---|---|---|---|---|---|")
    baseline = summarize_corpus_group(filter_records(tiny, experiment="baseline"), "legacy")
    if baseline:
        lines.append(
            f"| 100/0 | {baseline['mean_correct']:.4f} | {baseline['mean_incorrect']:.4f} "
            f"| {baseline['mean_delta']:+.4f} | N/A | {baseline['seeds_correct']}/{baseline['n']} |"
        )
    for prop in [0.5, 0.4, 0.3, 0.2, 0.1]:
        group = summarize_corpus_group(filter_prop(filter_records(tiny, experiment="mixed"), prop), "legacy")
        if not group:
            continue
        pct = int(prop * 100)
        lines.append(
            f"| {pct}/{100-pct} | {fmt_pm(group['mean_correct'], group['std_correct'])} "
            f"| {fmt_pm(group['mean_incorrect'], group['std_incorrect'])} "
            f"| {fmt_signed(group['mean_delta'], group['std_delta'])} "
            f"| {fmt_ci(group['ci_lo'], group['ci_hi'])} "
            f"| {group['seeds_correct']}/{group['n']} |"
        )

    full_random = [
        (prop, summarize_corpus_group(filter_prop(filter_records(tiny, experiment="mixed"), prop), "full"))
        for prop in [0.5, 0.1]
    ]
    if any(group for _, group in full_random):
        heading("Deterministic Full-Test Robustness Checks")
        lines.append("| Condition | Loss correct | Loss incorrect | ΔLoss | Seeds -> correct |")
        lines.append("|---|---|---|---|---|")
        for prop, group in full_random:
            if not group:
                continue
            pct = int(prop * 100)
            lines.append(
                f"| mixed {pct}/{100-pct} | {fmt_pm(group['mean_correct'], group['std_correct'])} "
                f"| {fmt_pm(group['mean_incorrect'], group['std_incorrect'])} "
                f"| {fmt_signed(group['mean_delta'], group['std_delta'])} "
                f"| {group['seeds_correct']}/{group['n']} |"
            )
        coherent_full = summarize_corpus_group(filter_prop(filter_records(tiny, experiment="coherent"), 0.5), "full")
        if coherent_full:
            lines.append(
                f"| coherent 50/50 | {fmt_pm(coherent_full['mean_correct'], coherent_full['std_correct'])} "
                f"| {fmt_pm(coherent_full['mean_incorrect'], coherent_full['std_incorrect'])} "
                f"| {fmt_signed(coherent_full['mean_delta'], coherent_full['std_delta'])} "
                f"| {coherent_full['seeds_correct']}/{coherent_full['n']} |"
            )

    heading("Coherence Spectrum at 50/50 (Paired)")
    lines.append("| Error type | Pair accuracy | Avg ΔLoss (paired) | Seed directions |")
    lines.append("|---|---|---|---|")
    for exp_name, label in [("mixed", "Random"), ("contradictory", "Contradictory"), ("coherent", "Coherent")]:
        group = summarize_paired_group(filter_prop(filter_records(tiny, experiment=exp_name), 0.5))
        if not group:
            continue
        lines.append(
            f"| {label} | {fmt_pm(group['mean_acc'], group['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(group['mean_delta'], group['std_delta'])} "
            f"| {group['seeds_correct']}/{group['n']} > 50% |"
        )

    heading("Paired Robustness Checks")
    lines.append("| Condition | Primary accuracy | Primary ΔLoss | Sum-NLL accuracy | Sum-NLL ΔLoss | Matched-length accuracy | Matched-length ΔLoss |")
    lines.append("|---|---|---|---|---|---|---|")
    robustness_specs = [
        ("mixed", 0.5, None, "math random 50/50"),
        ("mixed", 0.1, None, "math random 10/90"),
        ("coherent", 0.5, None, "math coherent 50/50"),
        ("world_random", 0.5, None, "world random 50/50"),
        ("world_coherent", 0.5, None, "world coherent 50/50"),
    ]
    for experiment, prop, variant, label in robustness_specs:
        subset = filter_prop(filter_records(tiny, experiment=experiment), prop)
        if variant is not None:
            subset = [r for r in subset if r["variant"] == variant]
        primary = summarize_paired_group(subset)
        sum_metric = summarize_paired_group(
            subset,
            accuracy_field="paired_sum_accuracy",
            delta_field="paired_sum_delta",
        )
        matched_metric = summarize_paired_group(
            subset,
            accuracy_field="paired_length_matched_accuracy",
            delta_field="paired_length_matched_delta",
        )
        if not primary or not sum_metric or not matched_metric:
            continue
        lines.append(
            f"| {label} "
            f"| {fmt_pm(primary['mean_acc'], primary['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(primary['mean_delta'], primary['std_delta'])} "
            f"| {fmt_pm(sum_metric['mean_acc'], sum_metric['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(sum_metric['mean_delta'], sum_metric['std_delta'])} "
            f"| {fmt_pm(matched_metric['mean_acc'], matched_metric['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(matched_metric['mean_delta'], matched_metric['std_delta'])} |"
        )

    heading("Multi-Rule Matched Evaluation")
    lines.append("| N rules | Pair accuracy | Avg ΔLoss (paired) | Evaluation family |")
    lines.append("|---|---|---|---|")
    n1 = summarize_paired_group(filter_records(tiny, experiment="coherent"))
    if n1:
        # The matched N=1 baseline is stored in auxiliary files, so we do not infer it here.
        lines.append("| 1 | see `results_manifest.md` | see `results_manifest.md` | matched coherent baseline stored separately |")
    for n_rules in [2, 3, 5, 10]:
        group = summarize_paired_group([r for r in tiny if r["experiment"] == "multirule" and r["variant"] == n_rules])
        if not group:
            continue
        lines.append(
            f"| {n_rules} | {fmt_pm(group['mean_acc'], group['std_acc'], decimals=1, pct=True)} "
            f"| {fmt_signed(group['mean_delta'], group['std_delta'])} | matched (`eval_paired_matched.json`) |"
        )

    heading("Scaling Coverage")
    lines.append("| Condition | tiny | small | medium | large |")
    lines.append("|---|---|---|---|---|")
    random_counts = []
    coherent_counts = []
    for size in ["tiny", "small", "medium", "large"]:
        random_counts.append(str(len(filter_prop(filter_records(records, experiment="mixed", model_size=size), 0.5))))
        coherent_counts.append(str(len(filter_prop(filter_records(records, experiment="coherent", model_size=size), 0.5))))
    lines.append(f"| random 50/50 paired seeds | {' | '.join(random_counts)} |")
    lines.append(f"| coherent 50/50 paired seeds | {' | '.join(coherent_counts)} |")

    return "\n".join(lines).strip() + "\n"


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
    write_csv(records, args.output_csv)

    tables = generate_tables(records)
    args.output_tables.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tables, "w") as f:
        f.write(tables)
    print(f"Wrote tables to {args.output_tables}")
    print("\n" + "=" * 70)
    print(tables)


if __name__ == "__main__":
    main()
