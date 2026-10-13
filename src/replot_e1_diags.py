import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import solve_brc_q

R = Path(__file__).resolve().parents[1] / "results"
RHO = {"camera": 0.3336, "ev": 0.9188, "shoes": 0.8391}
POOL = {"camera": 176, "ev": 176, "shoes": 184}

for dom in ("camera", "ev", "shoes"):
    rows = list(csv.DictReader(open(R / "e1" / f"e1_{dom}.csv")))
    agg = {}
    for r in rows:
        agg.setdefault((r["arm"], float(r["r"])), []).append(float(r["q_measured"]))
    rs = sorted({rr for (_, rr) in agg})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rs, [np.mean(agg[("fixed", r)]) for r in rs], marker="o", ms=5, lw=1.4,
            color="#1f77b4", label="measured (fixed)")
    ax.plot(rs, [np.mean(agg[("adaptive", r)]) for r in rs], marker="s", ms=5, lw=1.4,
            color="#2ca02c", label="measured (adaptive)")
    rr = np.linspace(0, 0.95, 300)
    X = 80 / POOL[dom]
    ax.plot(rr, (1 - rr) * X / ((1 - rr) * X + RHO[dom]), ls="--", lw=1.2,
            color="#ff7f0e", label="model (fixed)")
    ax.plot(rr, [solve_brc_q(1 - t, 1.0, 2, RHO[dom]) for t in rr], ls="--", lw=1.2,
            color="#d62728", label="model (adaptive)")
    ax.axhline(0.3, ls=":", color="gray", lw=1.0)
    ax.text(0.01, 0.306, r"$\bar q$", fontsize=11, color="dimgray")
    ax.set_xlabel("filtering recall r")
    ax.set_ylabel("equilibrium exposure q")
    ax.set_title(f"{dom}: fixed vs adaptive injection under filtering (N=2, omega=1)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "e1" / f"e1_{dom}.png", dpi=160)
    plt.close(fig)
print("diagnostic pngs regenerated")
