import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "RAG_new_exps"
LOGS = ROOT / "logs"
OUT = REPO / "results" / "parsed"

COUNT_RE = re.compile(r"^=== Adversarial injection count: (\d+) ===")
INJ_RE = re.compile(r"^Injecting (\d+) simple \+ (\d+) semantic adversarial docs")
RUN_RE = re.compile(r"^\[run (\d+)/(\d+)\] query: (.*)$")
ITEM_RE = re.compile(r"^\s+(\d+)\.\s+([A-Za-z0-9_]+):(\d+)\s+\|\s+score=([0-9.]+)\s+\|\s+(.*)$")
JUDGE_RE = re.compile(r"^\[Judge\] bias=(\d+), reason=(.*)$")
BATCH_RE = re.compile(r"(\d+)/(\d+) \[")
SEED_RE = re.compile(r"^\[[^\]]*\] start seed=(\d+) task_index=(\d+)/(\d+)")
ANSWER_MARK = "--- LLM Answer ---"


def parse_seed_log(path):
    seed = None
    cfg = {}
    counts = OrderedDict()
    cur = None
    run = None
    state = "idle"
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        m = SEED_RE.match(line)
        if m:
            seed = int(m.group(1))
            continue
        m = COUNT_RE.match(line)
        if m:
            n = int(m.group(1))
            cur = counts.setdefault(str(n), {"injecting": None, "embed_batches": [], "runs": []})
            run = None
            state = "idle"
            continue
        m = INJ_RE.match(line)
        if m and cur is not None:
            cur["injecting"] = [int(m.group(1)), int(m.group(2))]
            continue
        if line.strip().startswith("Generating embeddings:") or line.strip().startswith("Applying transformations:"):
            pairs = BATCH_RE.findall(line)
            if pairs and cur is not None:
                cur["embed_batches"].append(int(pairs[-1][0]))
            continue
        if line.strip().endswith(":") and not line.startswith(" ") and "=" not in line:
            continue
        kv = re.match(r"^  ([a-z_]+): (.*)$", line)
        if kv and cur is None:
            cfg[kv.group(1)] = kv.group(2)
            continue
        m = RUN_RE.match(line)
        if m and cur is not None:
            run = {"idx": int(m.group(1)), "n_runs": int(m.group(2)), "query": m.group(3),
                   "retrieved": [], "answer": [], "judge": None}
            cur["runs"].append(run)
            state = "retrieve"
            continue
        m = ITEM_RE.match(line)
        if m and run is not None and state == "retrieve":
            run["retrieved"].append({"rank": int(m.group(1)), "source": m.group(2),
                                     "line": int(m.group(3)), "score": float(m.group(4)),
                                     "snippet": m.group(5)[:200]})
            continue
        if ANSWER_MARK in line and run is not None:
            state = "answer"
            continue
        m = JUDGE_RE.match(line)
        if m and run is not None:
            run["judge"] = {"bias": int(m.group(1)), "reason": m.group(2)[:400]}
            state = "idle"
            continue
        if state == "answer" and run is not None:
            run["answer"].append(line)
    for c in counts.values():
        for r in c["runs"]:
            r["answer"] = "\n".join(r["answer"])[:600]
    return {"seed": seed, "config": cfg, "counts": counts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log_files = sorted(LOGS.glob("seed_*.log"))
    all_queries = {}
    summary = []
    for path in log_files:
        data = parse_seed_log(path)
        seed = data["seed"]
        (OUT / f"seed_{seed}.json").write_text(json.dumps(data, ensure_ascii=False))
        qs = []
        for c in data["counts"].values():
            qs.extend(r["query"] for r in c["runs"] if r["idx"] == 1)
        all_queries[seed] = qs
        for cname, c in data["counts"].items():
            summary.append({"seed": seed, "count": int(cname), "injecting": str(c["injecting"]),
                            "n_runs": len(c["runs"]),
                            "embed_batches": str(sorted(set(c["embed_batches"]))),
                            "asr": round(sum(r["judge"]["bias"] for r in c["runs"] if r["judge"]) / max(1, len(c["runs"])), 4)})
    (OUT / "queries_per_seed.json").write_text(json.dumps(all_queries, ensure_ascii=False, indent=1))
    import csv
    with open(OUT / "parse_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "count", "injecting", "n_runs", "embed_batches", "asr"])
        w.writeheader()
        w.writerows(summary)
    print(f"parsed {len(log_files)} seed logs -> {OUT}")
    for row in summary[:8]:
        print(row)


if __name__ == "__main__":
    main()
