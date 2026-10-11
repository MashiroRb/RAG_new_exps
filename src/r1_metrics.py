import csv
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "RAG_new_exps" / "results" / "r4" / "per_query_camera_ratio0.1.csv"
OUTD = ROOT / "RAG_new_exps" / "results" / "r1"
METRICS = ["q_run", "hit", "k_q", "rr_weighted", "score_weighted"]


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks = {}
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    s = sum(ranks[k] for k, (_, l) in enumerate(pairs) if l == 1)
    return (s - npos * (npos + 1) / 2) / (npos * nneg)


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            m = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = m
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)) * sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / den if den > 0 else float("nan")


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(IN)))
    for r in rows:
        for m in METRICS:
            r[m] = float(r[m])
        r["bias"] = int(r["bias"])
        r["seed"] = int(r["seed"])
        r["adv_count"] = int(r["adv_count"])

    summary = []
    for subset, name in [(rows, "all"), ([r for r in rows if r["adv_count"] > 0], "positive_injection_only")]:
        for m in METRICS:
            aucs = []
            rhos = []
            for s in sorted(set(r["seed"] for r in subset)):
                rs = [r for r in subset if r["seed"] == s]
                aucs.append(auc([r[m] for r in rs], [r["bias"] for r in rs]))
                if len(set(r[m] for r in rs)) > 1:
                    rhos.append(spearman([r[m] for r in rs], [r["bias"] for r in rs]))
            pooled = auc([r[m] for r in subset], [r["bias"] for r in subset])
            import numpy as np
            seeds = sorted(set(r["seed"] for r in subset))
            by_seed = {s: np.array([[r[m], r["bias"]] for r in subset if r["seed"] == s]) for s in seeds}
            rng = np.random.default_rng(2026)
            boot = []
            for _ in range(2000):
                parts = []
                for s in seeds:
                    src = by_seed[s]
                    idx = rng.integers(0, len(src), len(src))
                    parts.append(src[idx])
                arr = np.vstack(parts)
                boot.append(auc(list(arr[:, 0]), [int(v) for v in arr[:, 1]]))
            boot.sort()
            lo = boot[int(0.025 * len(boot))]
            hi = boot[int(0.975 * len(boot))]
            summary.append({"subset": name, "metric": m,
                            "auc_pooled": round(pooled, 4),
                            "auc_seed_mean": round(sum(aucs) / len(aucs), 4),
                            "auc_seed_std": round(math.sqrt(sum((a - sum(aucs) / len(aucs)) ** 2 for a in aucs) / len(aucs)), 4),
                            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                            "spearman_mean": round(sum(rhos) / len(rhos), 4) if rhos else ""})
    with open(OUTD / "single_condition_camera_ratio0.1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "metric", "auc_pooled", "auc_seed_mean",
                                          "auc_seed_std", "ci95_lo", "ci95_hi", "spearman_mean"])
        w.writeheader()
        w.writerows(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
