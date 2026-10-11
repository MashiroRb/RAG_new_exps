# RAG_new_exps — Governance-Model Validation Experiments

Offline (no-LLM-API) experiments validating the modeling section of
"When Retrieval Becomes Strategic" against reviewer criticism. All phases
run on CPU only, reusing the existing corpora, injected documents, query
pools and deterministic retrieval pipeline (bge-small-en-v1.5 + chunk 512/64
+ FAISS exact L2, top-k).

## Phases

| script | purpose |
|---|---|
| `src/parse_logs.py` | parse the five seed logs of the re-run camera/ratio0.1/k10 experiment: query pool, per-query retrieval lists (source:line, score, rank), injection composition, judge verdicts |
| `src/infer_injection.py` | recover the per-(seed,count) injected-document selection rule from observed retrieved ids |
| `src/replay_alignment.py` | offline replay of retrieval for camera/ratio0.1/k10 and comparison against `new_result` / `expb_result` exposure (alignment report) |
| `src/r1_metrics.py` | replay all 9 historical conditions (3 domains x normal/ratio0.1/top5) and compare five candidate exposure proxies (count share q, hit rate, adversarial count k_q, reciprocal-rank position weighting, score weighting) with cross-condition distances, held-out prediction and seed-level bootstrap |
| `src/r2_kappa.py` | selection effect: embedding-based detectors on simple vs CPA adversarial pools, survival vs retrievability, kappa estimate and nominal-recall correction |
| `src/r3_grid.py` | grid-side (dataA + queryA): replay of the 121 grid, filtering-response experiment (fixed vs adaptive injection on the q–r plane), authentic-mass (rho) lever, N-comparative statics |
| `src/r4_anchor.py` | new-generator anchor: exposure→ASR monotonicity within the re-run camera logs (retrieval-side unchanged, ASR shift isolated) |

## Layout

- `results/` — parsed logs, alignment reports, analysis outputs (auto-pushed)
- `logs/` — phase run logs (auto-pushed by `scripts/sync_loop.sh`)
- Data corpora live outside the repo (`../dataA`, `../dataB`, `../logs`, `../new_result`, `../expb_result`, `../expa_result.csv`, `../queryA.py`, `../queryB.py`).

## Run

```
bash scripts/run_all.sh          # sequential pipeline, status -> results/status.txt
bash scripts/sync_loop.sh 120    # periodic commit+push of logs+results
```
