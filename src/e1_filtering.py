import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import (Embedder, VecCache, build_index, exposure_of, format_text,
                        load_corpus, load_pool, nodes_of, r_star, solve_brc_q)

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "e1"
SEEDS = [101, 202, 303]
K = 10
OMEGA = 1.0
QBAR = 0.3


def load_queries(domain):
    src = (ROOT / "queryB.py").read_text()
    name = {"ev": "DEFAULT_QUERIES_EV", "camera": "DEFAULT_QUERIES_CAMERA", "shoes": "DEFAULT_QUERIES_SHOES"}[domain]
    import re
    block = src.split(name + " = [")[1].split("]")[0]
    return re.findall(r'"([^"]+)"', block)


def per_query_metrics(inj_rows, nodes):
    rows = []
    for qi, row in enumerate(inj_rows):
        inj_ranks = [pos + 1 for pos, (_, is_inj) in enumerate(row) if is_inj]
        n_inj = len(inj_ranks)
        dists = None
        rows.append({"q_idx": qi, "n_inj": n_inj, "q_run": n_inj / K,
                     "hit": 1.0 if n_inj > 0 else 0.0,
                     "rr_weighted": sum(1.0 / r for r in inj_ranks) / K})
    return rows


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "camera"
    OUTD.mkdir(parents=True, exist_ok=True)
    fit = json.load(open(ROOT / "RAG_new_exps" / "results" / "e0" / f"e0_fit_{domain}.json"))
    rho = fit["rho_hat"]
    half_docs = {"camera": 88, "ev": 88, "shoes": 92}[domain]
    corpus = load_corpus(domain)
    pool = load_pool(domain, "semantic")
    half = len(pool) // 2
    provA, provB = pool[:half], pool[half:]
    queries = load_queries(domain)
    embedder = Embedder(pooling="mean")
    cache = VecCache(embedder)
    corpus_nodes = nodes_of(corpus)
    qvecs = embedder.encode(queries)

    base_nA, base_nB = 40, 40
    r_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    rstar = r_star(OMEGA, 2, rho, QBAR)
    r_grid_sorted = sorted(set(r_grid + [round(rstar, 3)]))
    print(f"rho={rho} r*={rstar:.4f}")

    rows = []
    pq_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        permA = rng.permutation(len(provA))
        permB = rng.permutation(len(provB))
        baseA = [provA[i] for i in sorted(permA[:base_nA])]
        baseB = [provB[i] for i in sorted(permB[:base_nB])]
        for r in r_grid_sorted:
            eta = 1.0 - r
            X_norm = (base_nA + base_nB) / (2 * half_docs)
            q_mech = eta * X_norm / (eta * X_norm + rho)
            keep = rng.random(base_nA + base_nB) >= r
            keptA = [rec for rec, kp in zip(baseA, keep[:base_nA]) if kp]
            keptB = [rec for rec, kp in zip(baseB, keep[base_nA:]) if kp]
            inj = nodes_of(keptA + keptB)
            index, n_nodes = build_index(corpus_nodes + inj, cache)
            q_meas, inj_rows = exposure_of(index, len(corpus_nodes), qvecs, K)
            rows.append({"seed": seed, "arm": "fixed", "r": r, "n_injected_docs": len(keptA) + len(keptB),
                         "q_measured": round(q_meas, 4), "q_pred": round(q_mech, 4),
                         "abs_err": round(abs(q_meas - q_mech), 4)})
            for m in per_query_metrics(inj_rows, inj):
                pq_rows.append({"seed": seed, "arm": "fixed", "r": r, **m})
            q_star = solve_brc_q(eta, OMEGA, 2, rho)
            x_star = q_star * rho / (eta * 2 * (1 - q_star)) if q_star > 0 else 0.0
            n_star = int(round(x_star * 2 * half_docs))
            n_star = min(n_star, half)
            selA = [provA[i] for i in sorted(permA[:n_star])]
            selB = [provB[i] for i in sorted(permB[:n_star])]
            keep2 = rng.random(2 * n_star) >= r
            keptA2 = [rec for rec, kp in zip(selA, keep2[:n_star]) if kp]
            keptB2 = [rec for rec, kp in zip(selB, keep2[n_star:]) if kp]
            inj2 = nodes_of(keptA2 + keptB2)
            index2, _ = build_index(corpus_nodes + inj2, cache)
            q_adapt, inj_rows2 = exposure_of(index2, len(corpus_nodes), qvecs, K)
            rows.append({"seed": seed, "arm": "adaptive", "r": r, "n_injected_docs": len(keptA2) + len(keptB2),
                         "n_planned": 2 * n_star,
                         "q_measured": round(q_adapt, 4), "q_pred": round(q_star, 4),
                         "abs_err": round(abs(q_adapt - q_star), 4)})
            for m in per_query_metrics(inj_rows2, inj2):
                pq_rows.append({"seed": seed, "arm": "adaptive", "r": r, **m})
            print(f"seed {seed} r={r:.3f} fixed q={q_meas:.4f} (mech {q_mech:.4f}) | adaptive planned={2*n_star} kept={len(keptA2)+len(keptB2)} q={q_adapt:.4f} (BRC {q_star:.4f})", flush=True)
    with open(OUTD / f"e1_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "r", "n_injected_docs", "n_planned", "q_measured", "q_pred", "abs_err"])
        w.writeheader()
        w.writerows(rows)
    with open(OUTD / f"e1_{domain}_perquery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "r", "q_idx", "n_inj", "q_run", "hit", "rr_weighted"])
        w.writeheader()
        w.writerows(pq_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, pred_col, marker in [("fixed", "q_pred", "o"), ("adaptive", "q_pred", "s")]:
        pts = {}
        for r in r_grid_sorted:
            vals = [row["q_measured"] for row in rows if row["arm"] == arm and abs(row["r"] - r) < 1e-9]
            preds = [row["q_pred"] for row in rows if row["arm"] == arm and abs(row["r"] - r) < 1e-9]
            if vals:
                pts[r] = (np.mean(vals), np.mean(preds))
        rs = sorted(pts)
        ax.plot(rs, [pts[r][0] for r in rs], marker=marker, ms=5, lw=1.4, label=f"measured ({arm})")
        ax.plot(rs, [pts[r][1] for r in rs], ls="--", lw=1.2, label=f"model ({arm})")
    ax.axhline(QBAR, color="gray", ls=":", lw=1)
    ax.text(0.01, QBAR, r"$\bar q$", va="bottom")
    ax.set_xlabel("filtering recall r")
    ax.set_ylabel("equilibrium exposure q")
    ax.set_title(f"{domain}: fixed vs adaptive injection under filtering (N=2, omega=1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTD / f"e1_{domain}.png", dpi=160)
    print("E1 done ->", OUTD)


if __name__ == "__main__":
    main()
