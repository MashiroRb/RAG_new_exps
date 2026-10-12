import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "RAG_new_exps" / "results"
FIGD = ROOT / "paper" / "acl" / "figures"
FIGD.mkdir(parents=True, exist_ok=True)
DOMAINS = ("camera", "ev", "shoes")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

BLUE = "#4682B4"
ORANGE = "#E1812C"


def seed_means(rows, key_fields, val_field):
    agg = defaultdict(list)
    for r in rows:
        agg[tuple(r[k] for k in key_fields)].append(float(r[val_field]))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def style_axis(ax, title):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.plot([0, 1], [0, 1], ls=(0, (6, 3)), color="black", lw=2.0, zorder=1,
            label="Ideal fit")
    ax.grid(True, ls=":", lw=0.6, color="gray", alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=9)


def stats_box(ax, text):
    ax.text(0.035, 0.965, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=11.5, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="black", lw=1.0))


def legend_box(ax, loc):
    leg = ax.legend(loc=loc, frameon=True, framealpha=1.0, fancybox=False,
                    edgecolor="black", handlelength=2.2, borderpad=0.55,
                    labelspacing=0.45)
    leg.get_frame().set_linewidth(0.9)


def e1_figures(stats):
    for dom in DOMAINS:
        rows = list(csv.DictReader(open(R / "e1" / f"e1_{dom}.csv")))
        fit = json.load(open(R / "e0" / f"e0_fit_{dom}.json"))
        fig, ax = plt.subplots(figsize=(4.3, 3.9))
        arms = [("fixed", BLUE, "o", "Fixed injection"),
                ("adaptive", ORANGE, "s", "Strategic injection")]
        arm_err = {}
        for arm, col, mk, lab in arms:
            sel = [r for r in rows if r["arm"] == arm]
            arm_err[arm] = float(np.mean([float(r["abs_err"]) for r in sel]))
            ax.scatter([float(r["q_measured"]) for r in sel],
                       [float(r["q_pred"]) for r in sel],
                       s=42, marker=mk, facecolor=col, edgecolor="white",
                       linewidth=0.6, alpha=0.92, zorder=3, label=lab)
        rstar = {"camera": 0.80, "ev": 0.45, "shoes": 0.50}[dom]
        stats_box(ax, f"$\\hat\\rho = {fit['rho_hat']:.3f}$\n"
                      f"$r^\\star = {rstar:.2f}$\n"
                      f"MAE fixed $= {arm_err['fixed']:.3f}$\n"
                      f"MAE strat. $= {arm_err['adaptive']:.3f}$")
        style_axis(ax, f"FAISS/Dense Filtering Response ({dom})")
        ax.set_xlabel("Observed exposure $q$")
        ax.set_ylabel("Model-predicted exposure $\\hat{q}$")
        legend_box(ax, "lower right")
        fig.tight_layout()
        fig.savefig(FIGD / f"e1_filtering_{dom}.pdf")
        plt.close(fig)
        stats[f"e1_{dom}"] = {"arm_err": arm_err, "rstar": rstar}


def e3_figures(stats):
    for dom in DOMAINS:
        rows = list(csv.DictReader(open(R / "e3" / f"e3_{dom}.csv")))
        fig, ax = plt.subplots(figsize=(4.3, 3.9))
        eq = [r for r in rows if r["arm"] == "equilibrium_w*"]
        om = [r for r in rows if r["arm"].startswith("omega_")]
        capped_err = []
        for rows_, col, mk, lab in [(eq, BLUE, "o", "Equilibrium ($N$-sweep)"),
                                    (om, ORANGE, "s", "Benefit-ratio sweep")]:
            cap_x, cap_y, unc_x, unc_y = [], [], [], []
            for r in rows_:
                is_cap = r.get("capped") == "1"
                tgt = cap_x if is_cap else unc_x
                tgt2 = cap_y if is_cap else unc_y
                tgt.append(float(r["q_measured"]))
                tgt2.append(float(r["q_pred"]))
                if is_cap:
                    capped_err.append(abs(float(r["q_measured"]) - float(r["q_pred"])))
            ax.scatter(unc_x, unc_y, s=42, marker=mk, facecolor=col,
                       edgecolor="white", linewidth=0.6, alpha=0.92,
                       zorder=3, label=lab)
            if cap_x:
                lab = "Capped (pool limit)" if not hasattr(e3_figures, "_cap_labeled") else None
                e3_figures._cap_labeled = True
                ax.scatter(cap_x, cap_y, s=46, marker=mk, facecolor="white",
                           edgecolor=col, linewidth=1.2, alpha=0.95,
                           zorder=3, label=lab)
        err_eq = float(np.mean([abs(float(r["q_measured"]) - float(r["q_pred"]))
                                for r in eq if r.get("capped") != "1"]))
        err_om = float(np.mean([abs(float(r["q_measured"]) - float(r["q_pred"]))
                                for r in om if r.get("capped") != "1"]))
        om_star = {"camera": 0.148, "ev": 1.126, "shoes": 0.939}[dom]
        stats_box(ax, f"$\\omega^\\star = {om_star:.3f}$\n"
                      f"MAE $N$-sweep $= {err_eq:.3f}$\n"
                      f"MAE $\\omega$-sweep $= {err_om:.3f}$")
        style_axis(ax, f"FAISS/Dense Provider Competition ({dom})")
        ax.set_xlabel("Observed exposure $q$")
        ax.set_ylabel("Model-predicted exposure $\\hat{q}$")
        legend_box(ax, "lower right")
        fig.tight_layout()
        fig.savefig(FIGD / f"e3_nstatic_{dom}.pdf")
        plt.close(fig)
        stats[f"e3_{dom}"] = {"err_eq_uncapped": err_eq, "err_om_uncapped": err_om}


def main():
    stats = {}
    e1_figures(stats)
    e3_figures(stats)
    for dom in DOMAINS:
        eq = [r for r in csv.DictReader(open(R / "e3" / f"e3_{dom}.csv"))
              if r["arm"] == "equilibrium_w*"]
        agg = defaultdict(list)
        for r in eq:
            agg[int(r["N"])].append((float(r["x_star_docs"]), float(r["q_measured"]),
                                     float(r["q_pred"]), r.get("capped") == "1"))
        stats[f"e3_nsweep_{dom}"] = {
            N: {"x_star": float(np.mean([a for a, _, _, _ in v])),
                "q_meas": float(np.mean([b for _, b, _, _ in v])),
                "q_pred": float(np.mean([c for _, _, c, _ in v])),
                "capped": all(d for _, _, _, d in v)}
            for N, v in sorted(agg.items())}
    out = R / "paper_fig_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
