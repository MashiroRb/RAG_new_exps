import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import Embedder, format_text, load_corpus, load_pool, nodes_of

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "e0"
SEEDS = [101, 202, 303]
K = 10


def load_queries(domain):
    src = (ROOT / "queryB.py").read_text()
    name = {"ev": "DEFAULT_QUERIES_EV", "camera": "DEFAULT_QUERIES_CAMERA", "shoes": "DEFAULT_QUERIES_SHOES"}[domain]
    block = src.split(name + " = [")[1].split("]")[0]
    import re
    return re.findall(r'"([^"]+)"', block)


def chunk_texts(texts, size=512, overlap=64):
    from llama_index.core.node_parser import SentenceSplitter
    sp = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for t in texts:
        out.extend(sp.split_text(t))
    return out


def build_index(corpus_nodes, inj_nodes, embedder, vec_cache):
    texts = [t for _, _, t in corpus_nodes] + [t for _, _, t in inj_nodes]
    miss = [t for t in texts if t not in vec_cache]
    if miss:
        vecs = embedder.encode(miss)
        for t, v in zip(miss, vecs):
            vec_cache[t] = v
    mat = np.vstack([vec_cache[t] for t in texts])
    import faiss
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    return index


def exposure_of(index, qvecs, inj_node_ids, k=10):
    d, i = index.search(qvecs, k)
    inj_set = set(inj_node_ids)
    counts = [sum(1 for j in row if j in inj_set) for row in i]
    return sum(counts) / (len(counts) * k), counts


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "camera"
    OUTD.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(domain)
    pool = load_pool(domain, "semantic")
    half = len(pool) // 2
    provA, provB = pool[:half], pool[half:]
    queries = load_queries(domain)
    embedder = Embedder(pooling="mean")
    vec_cache = {}

    corpus_nodes = []
    for rec in corpus:
        for ch in chunk_texts([format_text(rec)]):
            corpus_nodes.append((rec["_source"], rec["_line"], ch))
    qvecs = embedder.encode(queries)

    grid = [(a, b) for a, b in product([0, 10, 20, 40, 80], [0, 10, 20, 40, 80])]
    grid += [(2, 2), (5, 5), (10, 10), (5, 0), (0, 5)]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        permA = rng.permutation(len(provA))
        permB = rng.permutation(len(provB))
        for na, nb in grid:
            selA = [provA[i] for i in sorted(permA[:na])]
            selB = [provB[i] for i in sorted(permB[:nb])]
            inj_nodes = nodes_of(selA + selB)
            inj_ids = list(range(len(corpus_nodes), len(corpus_nodes) + len(inj_nodes)))
            index = build_index(corpus_nodes, inj_nodes, embedder, vec_cache)
            q_mean, counts = exposure_of(index, qvecs, inj_ids, K)
            rows.append({"domain": domain, "seed": seed, "n_A": na, "n_B": nb,
                         "X": (na + nb) / (2 * half), "exposure": round(q_mean, 4),
                         "n_inj_chunks": len(inj_nodes)})
            print(f"seed {seed} (nA={na},nB={nb}) X={(na+nb)/(2*half):.3f} q={q_mean:.4f}", flush=True)
    with open(OUTD / f"e0_grid_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "seed", "n_A", "n_B", "X", "exposure", "n_inj_chunks"])
        w.writeheader()
        w.writerows(rows)
    df = [r for r in rows if r["X"] > 0]
    xs = np.array([r["X"] for r in df])
    ys = np.array([r["exposure"] for r in df])
    res = least_squares(lambda p: ys - xs / (xs + p[0]), [0.2])
    rho = res.x[0]
    pred = xs / (xs + rho)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    low = [(r for r in df if r["X"] <= 0.25)]
    summary = {"domain": domain, "rho_hat": round(float(rho), 4), "r2": round(r2, 4),
               "mae": round(float(np.mean(np.abs(ys - pred))), 4),
               "rmse": round(float(np.sqrt(np.mean((ys - pred) ** 2))), 4),
               "n_points": len(df)}
    with open(OUTD / f"e0_fit_{domain}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E0 FIT:", summary)


if __name__ == "__main__":
    main()
