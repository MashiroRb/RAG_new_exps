import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PARSED = ROOT / "RAG_new_exps" / "results" / "parsed"
OUTD = ROOT / "RAG_new_exps" / "results" / "r4"
SEEDS = [11, 22, 33, 44, 55]
K = 10


def load_history():
    rows = []
    with open(ROOT / "expb_result" / "camera" / "multiseed_summary_mean_std_camera_ratio0.1.csv") as f:
        for r in csv.DictReader(f):
            rows.append({"adv_count": int(r["adv_count"]), "exposure": float(r["exposure"]),
                         "asr": float(r["asr"]), "asr_std": float(r["asr_std"])})
    return rows


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    per_query_rows = []
    for s in SEEDS:
        d = json.load(open(PARSED / f"seed_{s}.json"))
        curve = {}
        base = None
        for cn in sorted(d["counts"], key=int):
            runs = d["counts"][cn]["runs"]
            biases = [r["judge"]["bias"] for r in runs if r["judge"]]
            inj = [sum(1 for it in r["retrieved"] if it["source"] != "final_camera_reviews") for r in runs]
            q = sum(inj) / (len(inj) * K)
            asr = sum(biases) / len(biases)
            curve[int(cn)] = {"q": q, "asr": asr, "n": len(biases)}
            if int(cn) == 0:
                base = asr
            for r, n_inj in zip(runs, inj):
                if r["judge"] is None:
                    continue
                ranks = [it["rank"] for it in r["retrieved"] if it["source"] != "final_camera_reviews"]
                rr = sum(1.0 / rk for rk in ranks)
                dists = [it["score"] for it in r["retrieved"]]
                mx = max(dists) if dists else 1.0
                wsum = sum(math.exp(-(d - mx)) for d in dists)
                w_inj = sum(math.exp(-(it["score"] - mx)) for it in r["retrieved"] if it["source"] != "final_camera_reviews")
                per_query_rows.append({"seed": s, "adv_count": int(cn), "q_run": n_inj / K,
                                       "hit": 1.0 if n_inj > 0 else 0.0, "k_q": n_inj,
                                       "rr_weighted": rr / K, "score_weighted": (w_inj / wsum) if wsum > 0 else 0.0,
                                       "bias": r["judge"]["bias"]})
        per_seed[s] = curve
    with open(OUTD / "per_query_camera_ratio0.1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    mean_curve = []
    for cn in sorted(per_seed[11]):
        q = sum(per_seed[s][cn]["q"] for s in SEEDS) / len(SEEDS)
        a = sum(per_seed[s][cn]["asr"] for s in SEEDS) / len(SEEDS)
        mean_curve.append({"adv_count": cn, "exposure": q, "asr_new": a})
    base = mean_curve[0]["asr_new"]
    for row in mean_curve:
        row["delta_asr_new"] = row["asr_new"] - base

    hist = load_history()
    pairs = [(row["exposure"], row["delta_asr_new"]) for row in mean_curve]
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    def spearman(a, b):
        ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
        rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
        n = len(a)
        d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
        return 1 - 6 * d2 / (n * (n * n - 1))
    rho = spearman(xs, ys)
    violations = sum(1 for i in range(len(xs)) for j in range(len(xs)) if xs[i] < xs[j] and ys[i] > ys[j])
    hist_base = hist[0]["asr"]
    with open(OUTD / "anchor_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adv_count", "exposure_new", "asr_new", "delta_asr_new", "asr_hist", "delta_asr_hist"])
        for row, h in zip(mean_curve, hist):
            w.writerow([row["adv_count"], round(row["exposure"], 4), round(row["asr_new"], 4),
                        round(row["delta_asr_new"], 4), h["asr"], round(h["asr"] - hist_base, 4)])
        w.writerow([])
        w.writerow(["spearman_exposure_vs_deltaASR_new", round(rho, 4)])
        w.writerow(["monotonicity_violations_pairs", violations])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar([h["exposure"] for h in hist], [h["asr"] - hist_base for h in hist],
                yerr=[h["asr_std"] for h in hist], marker="o", ms=4, lw=1.2, capsize=2,
                label="historical generator (aggregated results)")
    ax.plot(xs, ys, marker="s", ms=4, lw=1.2, label="re-run generator (parsed logs)")
    ax.set_xlabel("retrieval exposure q")
    ax.set_ylabel(r"$\Delta$ASR vs clean baseline")
    ax.set_title("camera, ratio=0.1, k=10: exposure axis held fixed across generators")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTD / "r4_anchor_overlay.png", dpi=160)
    print("r4 done: spearman=%.3f violations=%d" % (rho, violations))


if __name__ == "__main__":
    main()
