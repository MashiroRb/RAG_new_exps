import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import Embedder, VecCache, build_index, exposure_of, format_text, load_corpus, load_pool, nodes_of

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "e2"
SEEDS = [101, 202, 303]
K = 10
HALF_DOCS = {"camera": 88, "ev": 88, "shoes": 92}


def load_queries(domain):
    import re
    src = (ROOT / "queryB.py").read_text()
    name = {"ev": "DEFAULT_QUERIES_EV", "camera": "DEFAULT_QUERIES_CAMERA", "shoes": "DEFAULT_QUERIES_SHOES"}[domain]
    block = src.split(name + " = [")[1].split("]")[0]
    return re.findall(r'"([^"]+)"', block)


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "camera"
    OUTD.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(domain)
    pool = load_pool(domain, "semantic")
    half = len(pool) // 2
    provA, provB = pool[:half], pool[half:]
    queries = load_queries(domain)
    embedder = Embedder(pooling="mean")
    cache = VecCache(embedder)
    qvecs = embedder.encode(queries)
    nA = nB = 40
    fracs = [0.25, 0.5, 1.0]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 777)
        perm_corpus = rng.permutation(len(corpus))
        permA = np.random.default_rng(seed).permutation(len(provA))
        permB = np.random.default_rng(seed + 1).permutation(len(provB))
        selA = [provA[i] for i in sorted(permA[:nA])]
        selB = [provB[i] for i in sorted(permB[:nB])]
        inj = nodes_of(selA + selB)
        for f in fracs:
            n_keep = int(round(f * len(corpus)))
            sub = [corpus[i] for i in sorted(perm_corpus[:n_keep])]
            corpus_nodes = nodes_of(sub)
            index, base_n = build_index(corpus_nodes + inj, cache)
            q_meas, _ = exposure_of(index, base_n, qvecs, K)
            X = (nA + nB) / (2 * HALF_DOCS[domain])
            rho_imp = X * (1 - q_meas) / q_meas if q_meas > 0 else float("nan")
            rows.append({"seed": seed, "frac": f, "n_corpus_docs": n_keep,
                         "X": round(X, 4), "q_measured": round(q_meas, 4),
                         "rho_implied": round(rho_imp, 4)})
            print(f"seed {seed} frac={f} q={q_meas:.4f} rho_implied={rho_imp:.4f}", flush=True)
    with open(OUTD / f"e2_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "frac", "n_corpus_docs", "X", "q_measured", "rho_implied"])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for frac in fracs:
        vals = [r["rho_implied"] for r in rows if r["frac"] == frac]
        summary[str(frac)] = round(float(np.mean(vals)), 4)
    base = summary.get("1.0")
    scaling = {k: round(v / base, 4) for k, v in summary.items()} if base else {}
    with open(OUTD / f"e2_{domain}_summary.json", "w") as f:
        json.dump({"rho_implied_mean": summary, "ratio_to_full": scaling}, f, indent=1)
    print("E2 summary:", summary, "ratios:", scaling)


if __name__ == "__main__":
    main()
