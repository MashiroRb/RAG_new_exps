import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "r1p"
METRICS = ["q_run", "hit", "rr_weighted"]


def variance_components(values, groups):
    groups = np.asarray(groups)
    values = np.asarray(values, dtype=float)
    mu = values.mean()
    ss_between = 0.0
    ss_within = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - mu) ** 2
        ss_within += ((v - v.mean()) ** 2).sum()
    ss_total = ((values - mu) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else float("nan")


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "camera"
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "RAG_new_exps" / "results" / "e1" / f"e1_{domain}_perquery.csv")))
    out = []
    for metric in METRICS:
        vals = np.array([float(r[metric]) for r in rows])
        conds = np.array([f'{r["arm"]}|{r["r"]}' for r in rows])
        seeds = np.array([r["seed"] for r in rows])
        icc_cond = variance_components(vals, conds)
        icc_seed = variance_components(vals, seeds)
        out.append({"metric": metric, "icc_condition": round(icc_cond, 4), "icc_seed": round(icc_seed, 4),
                    "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)})
    with open(OUTD / f"metric_stability_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "icc_condition", "icc_seed", "mean", "std"])
        w.writeheader()
        w.writerows(out)
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
