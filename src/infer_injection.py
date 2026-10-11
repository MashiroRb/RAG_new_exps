import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RAG_new_exps" / "results" / "parsed"


def load_seed(seed):
    return json.load(open(OUT / f"seed_{seed}.json"))


def observed(seed, count):
    d = load_seed(seed)
    c = d["counts"][str(count)]
    sem, sim = set(), set()
    for r in c["runs"]:
        for it in r["retrieved"]:
            if it["source"] == "comments_adversarial":
                sem.add(it["line"])
            elif "simple" in it["source"]:
                sim.add(it["line"])
    queries = [r["query"] for r in c["runs"]]
    return sem, sim, queries


SEM_POOL = 176


def candidates(seed, count, n, pool_size):
    import numpy as np
    import random
    out = {}
    for kind, rng in [
        ("pss", np.random.RandomState(seed)),
        ("drng", np.random.default_rng(seed)),
        ("pyrand", random.Random(seed)),
    ]:
        try:
            if kind == "pyrand":
                out[kind] = set(rng.sample(range(pool_size), n))
            else:
                out[kind] = set(rng.choice(pool_size, size=n, replace=False).tolist())
        except Exception:
            out[kind] = None
    variants = {}
    for off in [0, 1, count, seed + count, seed * 100 + count, count * 100 + seed]:
        import numpy as np
        import random
        variants[f"pss+{off}"] = set(np.random.RandomState(seed + off).choice(pool_size, size=n, replace=False).tolist())
        variants[f"drng+{off}"] = set(np.random.default_rng(seed + off).choice(pool_size, size=n, replace=False).tolist())
        variants[f"py+{off}"] = set(random.Random(seed + off).sample(range(pool_size), n))
        variants[f"py+{off}1"] = set(random.Random(seed + off + 1).sample(range(pool_size), n))
    out.update(variants)
    return out


def main():
    import numpy as np
    scores = {}
    for seed in [11, 22, 33, 44, 55]:
        sem, _, _ = observed(seed, 10)
        for name, cand in candidates(seed, 10, 9, SEM_POOL).items():
            if cand is None:
                continue
            scores.setdefault(name, 0)
            if sem <= cand:
                scores[name] += 1
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    for name, sc in ranked[:12]:
        print(f"{name}: {sc}/5 seeds fully covered")
    best = ranked[0][0] if ranked else None
    print("best candidate:", best)


if __name__ == "__main__":
    main()
