"""
Regenerate scale_heldout_runs.csv / scale_heldout_summary.csv from the raw
per-run JSON under results_cluster/results/. The previous pair of files
stopped at 71 rows and predates the family measurements (TODO.md); this
collector reads every scale run and labels which support each number was
taken against, because the same run directory can carry both a realized and
a family measurement and the two have different nulls (SCALING_LAW.md §1).

Usage:
    python scripts/collect_scale_heldout.py [--results results_cluster/results] \
        [--out-runs results_cluster/scale_heldout_runs.csv] \
        [--out-summary results_cluster/scale_heldout_summary.csv]
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

PARAMS_M = {
    "tiny": 3.5, "small": 12.4, "medium": 25.8, "large": 85.4,
    "xlarge": 201.9, "xxlarge": 413.7, "qwen3-0.6b": 417.6, "qwen3-1b": 473.6,
}

# filename -> support label; heldout.json depends on the run's condition
NAMED = {
    "heldout_family.json": "keyed_family",
    "mass_realized.json": "keyed_realized",
    "mass_family.json": "keyed_family",
    "heldout_random_family.json": "random_family",
    "heldout_coherent.json": "coherent",
}
CONDITION_DEFAULT = {"random": "random_family", "coherent": "coherent",
                     "keyed": "keyed_realized", "clean": None}

DIR_RE = re.compile(r"^(scale2?)_(random|coherent|keyed|clean)_([\w.-]+)_seed(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_cluster/results")
    ap.add_argument("--out-runs", default="results_cluster/scale_heldout_runs.csv")
    ap.add_argument("--out-summary", default="results_cluster/scale_heldout_summary.csv")
    a = ap.parse_args()

    rows = []
    for d in sorted(Path(a.results).iterdir()):
        m = DIR_RE.match(d.name)
        if not m:
            continue
        grid, cond, size, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        for f in sorted(d.glob("*.json")):
            if f.name in ("config.json", "training_log.json"):
                continue
            support = NAMED.get(f.name)
            if support is None and f.name == "heldout.json":
                support = CONDITION_DEFAULT[cond]
            if support is None:
                continue
            j = json.loads(f.read_text())
            rows.append({
                "grid": grid, "condition": cond, "size": size,
                "arch": "Qwen3" if size.startswith("qwen") else "GPT-2",
                "params_m": PARAMS_M.get(size, ""), "seed": seed,
                "support": support, "source_file": f.name,
                "n_problems": j.get("n_problems"),
                "mean_false_modes": j.get("mean_false_modes"),
                "pair_accuracy": j.get("pair_accuracy_mean_nll"),
                "aggregate_share": j.get("aggregate_share_mean"),
                "eqlen_share": j.get("equal_length_share_mean"),
                "eqlen_n": j.get("equal_length_n"),
                "length_delta": j.get("length_delta_mean"),
            })

    rows.sort(key=lambda r: (r["grid"], r["condition"], r["params_m"] or 0,
                             r["support"], r["seed"]))
    with open(a.out_runs, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault(
            (r["grid"], r["condition"], r["size"], r["support"]), []).append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return (statistics.mean(vals),
                statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    with open(a.out_summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid", "condition", "size", "params_m", "support", "n_seeds",
                    "mean_false_modes", "pair_acc_mean", "pair_acc_sd",
                    "mass_all_mean", "mass_all_sd",
                    "mass_eqlen_mean", "mass_eqlen_sd", "eqlen_n"])
        for (grid, cond, size, support), g in sorted(
                groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                PARAMS_M.get(kv[0][2], 0), kv[0][3])):
            pa, pa_sd = ms([r["pair_accuracy"] for r in g])
            sh, sh_sd = ms([r["aggregate_share"] for r in g])
            eq, eq_sd = ms([r["eqlen_share"] for r in g])
            fmt = lambda v, k=100: round(v * k, 2) if v is not None else ""
            w.writerow([grid, cond, size, PARAMS_M.get(size, ""), support, len(g),
                        round(statistics.mean(r["mean_false_modes"] for r in g), 2),
                        fmt(pa), fmt(pa_sd), fmt(sh), fmt(sh_sd), fmt(eq), fmt(eq_sd),
                        g[0]["eqlen_n"]])

    print(f"{len(rows)} rows -> {a.out_runs}")
    print(f"{len(groups)} cells -> {a.out_summary}")


if __name__ == "__main__":
    main()
