import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import Embedder, observed_injected, load_corpus, load_pool, format_text, retrieve_blocks

ROOT = Path(__file__).resolve().parents[2]
PARSED = ROOT / "RAG_new_exps" / "results" / "parsed"
OUTD = ROOT / "RAG_new_exps" / "results" / "alignment"
SEEDS = [11, 22, 33, 44, 55]


def make_chunker():
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    def chunk(text):
        return splitter.split_text(text)
    return chunk


def count_fingerprint(chunker, sep):
    recs = load_corpus("camera")
    total = sum(len(chunker(format_text(r, sep=sep))) for r in recs)
    return total


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    chunker = make_chunker()
    sep = sys.argv[1] if len(sys.argv) > 1 else "\n"
    n0 = count_fingerprint(chunker, sep)
    print(f"camera corpus chunk count (sep={sep!r}): {n0} (target 2048)")

    pool_sem = load_pool("camera", "semantic")
    pool_sim = load_pool("camera", "simple")

    embedder = Embedder(pooling="mean")
    rows_out = []
    max_err = 0.0
    exact_blocks = 0
    total_blocks = 0
    corpus_recs = load_corpus("camera")
    corpus_vec_ready = False
    for s in SEEDS:
        sj = PARSED / f"seed_{s}.json"
        d = json.load(open(sj))
        for cn in sorted(d["counts"], key=int):
            sem_ids, sim_ids = observed_injected(sj, cn)
            inj = [pool_sem[i - 1] for i in sem_ids] + [pool_sim[i - 1] for i in sim_ids]
            queries = [r["query"] for r in d["counts"][cn]["runs"]]
            nodes, dists, idxs = retrieve_blocks(corpus_recs, inj, queries, embedder, chunker, k=10)
            logged = []
            for r in d["counts"][cn]["runs"]:
                logged.append([(it["source"], it["line"]) for it in r["retrieved"]])
            block_err = []
            block_exact = 0
            for qi in range(len(queries)):
                got = [(nodes[j][0], nodes[j][1]) for j in idxs[qi]]
                exp = logged[qi]
                same = sum(1 for a, b in zip(got, exp) if a == b)
                n_inj_got = sum(1 for a in got if a[0] != "final_camera_reviews")
                n_inj_exp = sum(1 for a in exp if a[0] != "final_camera_reviews")
                block_err.append(abs(n_inj_got - n_inj_exp) / 10)
                if got == exp:
                    block_exact += 1
            total_blocks += 1
            exact_blocks += block_exact
            q_logged = sum(sum(1 for a in l if a[0] != "final_camera_reviews") for l in logged) / (len(queries) * 10)
            q_replay = sum(1 for qi in range(len(queries)) for a in [(nodes[j][0], nodes[j][1]) for j in idxs[qi]] if a[0] != "final_camera_reviews") / (len(queries) * 10)
            err = abs(q_replay - q_logged)
            max_err = max(max_err, err)
            rows_out.append({"seed": s, "adv_count": int(cn), "n_injected": len(inj),
                             "exposure_logged": round(q_logged, 4), "exposure_replayed": round(q_replay, 4),
                             "abs_err": round(err, 4), "exact_lists": block_exact, "n_queries": len(queries)})
            print(f"seed {s} count {cn}: logged={q_logged:.4f} replayed={q_replay:.4f} err={err:.4f} exact={block_exact}/{len(queries)}", flush=True)
    import csv
    with open(OUTD / f"alignment_camera_ratio0.1_sep{repr(sep)}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "adv_count", "n_injected", "exposure_logged",
                                          "exposure_replayed", "abs_err", "exact_lists", "n_queries"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"MAX block-level |err| = {max_err:.4f}; exact top-10 lists: {exact_blocks}/{total_blocks * 50}")


if __name__ == "__main__":
    main()
