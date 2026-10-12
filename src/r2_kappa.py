import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import Embedder, format_text, load_corpus, load_pool

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "r2"
K = 10


def load_queries(domain):
    src = (ROOT / "queryB.py").read_text()
    name = {"ev": "DEFAULT_QUERIES_EV", "camera": "DEFAULT_QUERIES_CAMERA", "shoes": "DEFAULT_QUERIES_SHOES"}[domain]
    block = src.split(name + " = [")[1].split("]")[0]
    return re.findall(r'"([^"]+)"', block)


from replay_lib import chunk_texts as _chunk_texts


def chunk_texts(texts, size=512, overlap=64):
    return _chunk_texts(texts, size=size, overlap=overlap)


def retrievability(domain, corpus_nodes, pool_nodes, queries, embedder):
    corpus_vecs = embedder.encode([t for _, _, t in corpus_nodes])
    pool_vecs = embedder.encode([t for _, _, t in pool_nodes])
    qvecs = embedder.encode(queries)
    allv = np.vstack([corpus_vecs, pool_vecs])
    n_cor = len(corpus_vecs)
    w = np.zeros(len(pool_nodes))
    for qi in range(len(queries)):
        d = np.linalg.norm(allv - qvecs[qi], axis=1)
        order = np.argsort(d, kind="stable")[:K]
        hits = {j - n_cor for j in order if j >= n_cor}
        for j in hits:
            w[j] += 1.0 / len(queries)
    return w, corpus_vecs, pool_vecs


def doc_embedding(embedder, recs):
    return embedder.encode([format_text(r)[:6000] for r in recs])


def kappa_analysis(doc_vecs, corpus_doc_vecs, w, recalls):
    sims = doc_vecs @ corpus_doc_vecs.T
    max_sim = sims.max(axis=1)
    mean_top5 = np.sort(sims, axis=1)[:, -5:].mean(axis=1)
    rows = []
    for det_name, det in [("max_sim", max_sim), ("density_top5", mean_top5)]:
        for r in recalls:
            tau = np.quantile(det, 1.0 - r)
            survive = det <= tau
            wsum = w.sum()
            if wsum <= 0:
                continue
            eta_eff = float((survive * w).sum() / wsum)
            kappa = eta_eff / (1.0 - r) if r < 1 else float("nan")
            ret = w > 0
            p_surv_ret = float(survive[ret].mean()) if ret.any() else float("nan")
            p_surv_unret = float(survive[~ret].mean()) if (~ret).any() else float("nan")
            rows.append({"detector": det_name, "nominal_recall": r, "kappa": round(kappa, 4),
                         "eta_eff": round(eta_eff, 4),
                         "p_survive_retrievable": round(p_surv_ret, 4),
                         "p_survive_unretrievable": round(p_surv_unret, 4)})
    return rows


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(pooling="mean")
    all_rows = []
    for domain in ["camera", "ev", "shoes"]:
        corpus = load_corpus(domain)
        simple = load_pool(domain, "simple")
        cpa = load_pool(domain, "semantic")
        queries = load_queries(domain)
        chunker = lambda texts: chunk_texts(texts)
        corpus_nodes = []
        for rec in corpus:
            for ch in chunker(format_text(rec)):
                corpus_nodes.append((rec["_source"], rec["_line"], ch))
        simple_nodes = []
        for rec in simple:
            for ch in chunker([format_text(rec)]):
                simple_nodes.append((rec["_source"], rec["_line"], ch))
        cpa_nodes = []
        for rec in cpa:
            for ch in chunker([format_text(rec)]):
                cpa_nodes.append((rec["_source"], rec["_line"], ch))
        pool_nodes = simple_nodes + cpa_nodes
        w, corpus_vecs, pool_vecs = retrievability(domain, corpus_nodes, pool_nodes, queries, embedder)
        n_s = len(simple_nodes)
        w_simple, w_cpa = w[:n_s], w[n_s:]
        dv_simple = pool_vecs[:n_s]
        dv_cpa = pool_vecs[n_s:]
        for typ, wv, dv in [("simple", w_simple, dv_simple), ("cpa", w_cpa, dv_cpa)]:
            rows = kappa_analysis(dv, corpus_vecs, wv, [0.2, 0.3, 0.5, 0.7, 0.9])
            for row in rows:
                row.update({"domain": domain, "doc_type": typ,
                            "retrievability_any": round(float((wv > 0).mean()), 4),
                            "retrievability_mean": round(float(wv.mean()), 4)})
            all_rows.extend(rows)
            print(domain, typ, "retrievable share:", round(float((wv > 0).mean()), 3), flush=True)
    with open(OUTD / "kappa_table.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["domain", "doc_type", "detector", "nominal_recall", "kappa",
                                            "eta_eff", "p_survive_retrievable", "p_survive_unretrievable",
                                            "retrievability_any", "retrievability_mean"])
        wtr.writeheader()
        wtr.writerows(all_rows)
    print("r2 done ->", OUTD / "kappa_table.csv")


if __name__ == "__main__":
    main()
