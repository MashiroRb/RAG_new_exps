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
    fracs = [0.25, 0.5, 1.0]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 777)
        perm_corpus = rng.permutation(len(corpus))
        permA = np.random.default_rng(seed).permutation(len(provA))
        permB = np.random.default_rng(seed + 1).permutation(len(provB))
        for arm, base in [("abs80", 40), ("scaled", None)]:
            for f in fracs:
                nA = base if base else max(1, int(round(40 * f)))
                selA = [provA[i] for i in sorted(permA[:nA])]
                selB = [provB[i] for i in sorted(permB[:nA])]
                inj = nodes_of(selA + selB)
                n_keep = int(round(f * len(corpus)))
                sub = [corpus[i] for i in perm_corpus[:n_keep]]
                corpus_nodes = nodes_of(sub)
                index, _ = build_index(corpus_nodes + inj, cache)
                q_meas, _ = exposure_of(index, len(corpus_nodes), qvecs, K)
                n_inj = len(inj)
                X = n_inj / (len(corpus_nodes) + n_inj)
                rho_imp = X * (1 - q_meas) / q_meas if q_meas > 0 else float("nan")
                rows.append({"seed": seed, "arm": arm, "frac": f, "n_corpus_docs": n_keep,
                             "corpus_chunks": len(corpus_nodes), "inj_chunks": n_inj,
                             "X": round(X, 4), "q_measured": round(q_meas, 4),
                             "rho_implied": round(rho_imp, 4),
                             "rho_per_frac": round(rho_imp / f, 4) if rho_imp == rho_imp else float("nan")})
                print(f"seed {seed} {arm} frac={f} q={q_meas:.4f} X={X:.4f} rho_implied={rho_imp:.4f} rho/f={rho_imp / f:.4f}", flush=True)
    with open(OUTD / f"e2_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "frac", "n_corpus_docs", "corpus_chunks", "inj_chunks", "X", "q_measured", "rho_implied", "rho_per_frac"])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for arm in ["abs80", "scaled"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        summ = {}
        pf = {}
        for frac in fracs:
            vals = [r["rho_implied"] for r in arm_rows if r["frac"] == frac]
            perfs = [r["rho_per_frac"] for r in arm_rows if r["frac"] == frac]
            summ[str(frac)] = round(float(np.mean(vals)), 4)
            pf[str(frac)] = round(float(np.mean(perfs)), 4)
        base = summ.get("1.0")
        scaling = {k: round(v / base, 4) for k, v in summ.items()} if base else {}
        summary[arm] = {"rho_implied_mean": summ, "ratio_to_full": scaling, "rho_per_frac_mean": pf}
    with open(OUTD / f"e2_{domain}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E2 summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
