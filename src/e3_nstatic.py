import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import Embedder, VecCache, build_index, exposure_of, load_corpus, load_pool, nodes_of, solve_brc_q

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "RAG_new_exps" / "results" / "e3"
SEEDS = [101, 202, 303]
K = 10
OMEGA = 1.0


def load_queries(domain):
    import re
    src = (ROOT / "queryB.py").read_text()
    name = {"ev": "DEFAULT_QUERIES_EV", "camera": "DEFAULT_QUERIES_CAMERA", "shoes": "DEFAULT_QUERIES_SHOES"}[domain]
    block = src.split(name + " = [")[1].split("]")[0]
    return re.findall(r'"([^"]+)"', block)


def asymmetric_equilibrium(c1, c2, eta, omega, rho):
    def focs(x):
        x1, x2 = x
        D = eta * (x1 + x2) + rho
        return [omega * eta * (eta * x2 + rho) / (D * D) - c1 * x1,
                omega * eta * (eta * x1 + rho) / (D * D) - c2 * x2]
    sol = root(focs, [0.3, 0.3], method="hybr")
    if not sol.success or min(sol.x) < 0:
        return None
    return sol.x


def provider_shares(inj_rows, base_n, offsets):
    counts = [0] * len(offsets[:-1])
    for row in inj_rows:
        for j, isin in row:
            if isin:
                idx = j - base_n
                for pi in range(len(offsets) - 1):
                    if offsets[pi] <= idx < offsets[pi + 1]:
                        counts[pi] += 1
                        break
    tot = sum(counts)
    return [c / tot for c in counts] if tot else [0.0] * len(counts)


def run_config(corpus_nodes, cache, qvecs, sel_recs, seed, arm, N, m, rho, scale_doc, extra=None):
    inj = nodes_of(sel_recs)
    offsets = [0]
    per_provider_chunks = []
    for rec in sel_recs:
        pass
    index, base_n = build_index(corpus_nodes + inj, cache)
    q_meas, inj_rows = exposure_of(index, base_n, qvecs, K)
    row = {"seed": seed, "arm": arm, "N": N, "m": m, "X_docs": len(sel_recs),
           "X": round(len(sel_recs) / scale_doc, 4), "q_measured": round(q_meas, 4)}
    if extra:
        row.update(extra)
    return row, inj, base_n, inj_rows


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "camera"
    OUTD.mkdir(parents=True, exist_ok=True)
    fit = json.loads(open(ROOT / "RAG_new_exps" / "results" / "e0" / f"e0_fit_{domain}.json").read())
    rho = fit["rho_hat"]
    corpus = load_corpus(domain)
    pool = load_pool(domain, "semantic")
    queries = load_queries(domain)
    embedder = Embedder(pooling="mean")
    cache = VecCache(embedder)
    corpus_nodes = nodes_of(corpus)
    qvecs = embedder.encode(queries)
    n_prov_total = 8
    part = len(pool) // n_prov_total
    provs = [pool[i * part:(i + 1) * part] for i in range(n_prov_total)]
    scale_doc = len(pool)
    rows = []
    for seed in SEEDS:
        perms = [np.random.default_rng(seed * 100 + i).permutation(len(p)) for i, p in enumerate(provs)]
        for N in [1, 2, 4, 8]:
            use = provs[:N]
            for m in [5, 10, 20]:
                sel = [p[i] for p, pm in zip(use, perms) for i in sorted(pm[:m])]
                inj = nodes_of(sel)
                offsets = [0]
                cnt = 0
                for p in use:
                    cnt += len(nodes_of(p[:0]) or [])
                    cnt_per = sum(1 for rec in sel if False)
                    break
                index, base_n = build_index(corpus_nodes + inj, cache)
                q_meas, inj_rows = exposure_of(index, base_n, qvecs, K)
                rows.append({"seed": seed, "arm": "equal_m", "N": N, "m": m,
                             "X_docs": N * m, "X": round(N * m / scale_doc, 4),
                             "q_measured": round(q_meas, 4)})
                print(f"seed {seed} equal N={N} m={m}: q={q_meas:.4f}", flush=True)
            q0 = solve_brc_q(1.0, OMEGA, N, rho)
            x_star = q0 * rho / (N * (1 - q0))
            n_each = int(round(x_star * scale_doc / N))
            n_each = max(1, min(n_each, part))
            sel = [p[i] for p, pm in zip(use, perms) for i in sorted(pm[:n_each])]
            inj = nodes_of(sel)
            offsets = []
            acc = 0
            for p, pm in zip(use, perms):
                subset = [p[i] for i in sorted(pm[:n_each])]
                acc += len(nodes_of(subset))
                offsets.append(acc)
            index, base_n = build_index(corpus_nodes + inj, cache)
            q_meas, inj_rows = exposure_of(index, base_n, qvecs, K)
            shares = provider_shares(inj_rows, base_n, offsets)
            rows.append({"seed": seed, "arm": "equilibrium_r0", "N": N, "m": n_each,
                         "X_docs": N * n_each, "X": round(N * n_each / scale_doc, 4),
                         "q_measured": round(q_meas, 4), "q_pred": round(q0, 4),
                         "x_star_docs": n_each,
                         "share_max": round(max(shares), 4), "share_min": round(min(shares), 4)})
            print(f"seed {seed} eq N={N}: x*={n_each} q={q_meas:.4f} vs q0={q0:.4f} shares[{max(shares):.2f},{min(shares):.2f}]", flush=True)
        sol = asymmetric_equilibrium(0.5, 2.0, 1.0, OMEGA, rho)
        if sol is not None:
            n1 = max(1, min(int(round(sol[0] * scale_doc / 2)), part))
            n2 = max(1, min(int(round(sol[1] * scale_doc / 2)), part))
            sel = [provs[0][i] for i in sorted(perms[0][:n1])] + [provs[1][i] for i in sorted(perms[1][:n2])]
            inj = nodes_of(sel)
            offsets = [0, len(nodes_of([provs[0][i] for i in sorted(perms[0][:n1])])), len(inj)]
            index, base_n = build_index(corpus_nodes + inj, cache)
            q_meas, inj_rows = exposure_of(index, base_n, qvecs, K)
            shares = provider_shares(inj_rows, base_n, offsets)
            rows.append({"seed": seed, "arm": "asymmetric_c(0.5,2)", "N": 2, "m": n1,
                         "X_docs": n1 + n2, "X": round((n1 + n2) / scale_doc, 4),
                         "q_measured": round(q_meas, 4),
                         "share_lowcost": round(shares[0], 4), "share_highcost": round(shares[1], 4),
                         "x_pred_low_docs": round(sol[0] * scale_doc / 2, 1),
                         "x_pred_high_docs": round(sol[1] * scale_doc / 2, 1)})
            print(f"seed {seed} asym: n=({n1},{n2}) q={q_meas:.4f} shares=({shares[0]:.2f},{shares[1]:.2f})", flush=True)
    with open(OUTD / f"e3_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "N", "m", "X_docs", "X", "q_measured", "q_pred",
                                          "x_star_docs", "share_max", "share_min", "share_lowcost",
                                          "share_highcost", "x_pred_low_docs", "x_pred_high_docs"])
        w.writeheader()
        w.writerows(rows)
    print("E3 done ->", OUTD)


if __name__ == "__main__":
    main()
