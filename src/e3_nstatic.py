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
Q_MID = 0.5


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
    sol = root(focs, [0.15, 0.05], method="hybr")
    if not sol.success or min(sol.x) < 0:
        return None
    return sol.x


def provider_shares(inj_rows, base_n, offsets):
    counts = [0] * (len(offsets) - 1)
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


def place(corpus_nodes, cache, qvecs, sel, base_count, K):
    inj = nodes_of(sel)
    index, _ = build_index(corpus_nodes + inj, cache)
    q_meas, inj_rows = exposure_of(index, base_count, qvecs, K)
    return inj, q_meas, inj_rows


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
    base_count = len(corpus_nodes)
    qvecs = embedder.encode(queries)
    omega_star = Q_MID * rho * rho / ((1 - Q_MID) ** 2 * (2 - Q_MID))
    scale_doc = len(pool)
    rows = []
    print(f"rho={rho:.4f} omega_star={omega_star:.4f}", flush=True)
    for seed in SEEDS:
        for N in [1, 2, 4, 8]:
            part = len(pool) // N
            provs = [pool[i * part:(i + 1) * part] for i in range(N)]
            perms = [np.random.default_rng(seed * 100 + i).permutation(len(p)) for i, p in enumerate(provs)]
            for m in [5, 10, 20]:
                sel = [p[i] for p, pm in zip(provs, perms) for i in sorted(pm[:m])]
                _, q_meas, _ = place(corpus_nodes, cache, qvecs, sel, base_count, K)
                X = N * m / scale_doc
                rows.append({"seed": seed, "arm": "equal_m", "N": N, "m": m,
                             "X_docs": N * m, "X": round(X, 4),
                             "q_measured": round(q_meas, 4), "q_pred": round(X / (X + rho), 4)})
                print(f"seed {seed} equal N={N} m={m}: q={q_meas:.4f} pred={X / (X + rho):.4f}", flush=True)
            for omega, tag in [(omega_star, "equilibrium_w*"), (1.0, "equilibrium_w1")]:
                if tag == "equilibrium_w1" and N not in (1, 2):
                    continue
                q0 = solve_brc_q(1.0, omega, N, rho)
                x_j = q0 * rho / (N * (1 - q0)) if q0 > 0 else 0.0
                n_each = int(round(x_j * scale_doc))
                capped = n_each > part
                n_use = min(n_each, part)
                sel = [p[i] for p, pm in zip(provs, perms) for i in sorted(pm[:n_use])]
                _, q_meas, _ = place(corpus_nodes, cache, qvecs, sel, base_count, K)
                rows.append({"seed": seed, "arm": tag, "N": N, "m": n_use,
                             "X_docs": N * n_use, "X": round(N * n_use / scale_doc, 4),
                             "q_measured": round(q_meas, 4), "q_pred": round(q0, 4),
                             "x_star_docs": n_each, "capped": int(capped)})
                print(f"seed {seed} {tag} N={N}: x*={n_each} used={n_use} capped={capped} q={q_meas:.4f} vs q0={q0:.4f}", flush=True)
        part2 = len(pool) // 2
        provs2 = [pool[:part2], pool[part2:2 * part2]]
        perms2 = [np.random.default_rng(seed * 100 + i).permutation(len(p)) for i, p in enumerate(provs2)]
        for omega in [0.5, 1.0, 2.0]:
            q0 = solve_brc_q(1.0, omega, 2, rho)
            x_j = q0 * rho / (2 * (1 - q0)) if q0 > 0 else 0.0
            n_each = int(round(x_j * scale_doc))
            capped = n_each > part2
            n_use = min(n_each, part2)
            sel = [provs2[0][i] for i in sorted(perms2[0][:n_use])] + [provs2[1][i] for i in sorted(perms2[1][:n_use])]
            _, q_meas, _ = place(corpus_nodes, cache, qvecs, sel, base_count, K)
            rows.append({"seed": seed, "arm": f"omega_{omega}", "N": 2, "m": n_use,
                         "X_docs": 2 * n_use, "X": round(2 * n_use / scale_doc, 4),
                         "q_measured": round(q_meas, 4), "q_pred": round(q0, 4),
                         "x_star_docs": n_each, "capped": int(capped)})
            print(f"seed {seed} omega={omega}: x*={n_each} used={n_use} capped={capped} q={q_meas:.4f} vs q0={q0:.4f}", flush=True)
        sol = asymmetric_equilibrium(0.5, 2.0, 1.0, omega_star, rho)
        if sol is not None:
            n1 = max(1, min(int(round(sol[0] * scale_doc)), part2))
            n2 = max(1, min(int(round(sol[1] * scale_doc)), part2))
            selA = [provs2[0][i] for i in sorted(perms2[0][:n1])]
            selB = [provs2[1][i] for i in sorted(perms2[1][:n2])]
            inj = nodes_of(selA + selB)
            offsets = [0, len(nodes_of(selA)), len(inj)]
            index, _ = build_index(corpus_nodes + inj, cache)
            q_meas, inj_rows = exposure_of(index, base_count, qvecs, K)
            shares = provider_shares(inj_rows, base_count, offsets)
            rows.append({"seed": seed, "arm": "asymmetric_c(0.5,2)", "N": 2, "m": n1,
                         "X_docs": n1 + n2, "X": round((n1 + n2) / scale_doc, 4),
                         "q_measured": round(q_meas, 4),
                         "share_lowcost": round(shares[0], 4), "share_highcost": round(shares[1], 4),
                         "x_pred_low_docs": round(sol[0] * scale_doc, 1),
                         "x_pred_high_docs": round(sol[1] * scale_doc, 1)})
            print(f"seed {seed} asym: n=({n1},{n2}) q={q_meas:.4f} shares=({shares[0]:.2f},{shares[1]:.2f})", flush=True)
    with open(OUTD / f"e3_{domain}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "N", "m", "X_docs", "X", "q_measured", "q_pred",
                                          "x_star_docs", "capped", "share_lowcost", "share_highcost",
                                          "x_pred_low_docs", "x_pred_high_docs"])
        w.writeheader()
        w.writerows(rows)
    print("E3 done ->", OUTD)


if __name__ == "__main__":
    main()
