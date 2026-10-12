import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_lib import solve_brc_q

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "RAG_new_exps" / "results"
FIGD = ROOT / "paper" / "acl" / "figures"
FIGD.mkdir(parents=True, exist_ok=True)
DOMAINS = ("camera", "ev", "shoes")
RHO = {"camera": 0.3336, "ev": 0.9188, "shoes": 0.8391}
RSTAR = {"camera": 0.800, "ev": 0.448, "shoes": 0.497}
OMEGA_STAR = {"camera": 0.148, "ev": 1.126, "shoes": 0.939}
QBAR = 0.3

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

BLUE = "#4682B4"
ORANGE = "#E1812C"


def seed_stats(rows, key_fields, val_field):
    agg = defaultdict(list)
    for r in rows:
        agg[tuple(r[k] for k in key_fields)].append(float(r[val_field]))
    keys = sorted(agg, key=lambda k: float(k[-1]))
    xs = np.array([float(k[-1]) for k in keys])
    means = np.array([np.mean(agg[k]) for k in keys])
    stds = np.array([np.std(agg[k], ddof=1) if len(agg[k]) > 1 else 0.0 for k in keys])
    return xs, means, stds


def base_axis(ax, xlabel, ylabel, title):
    ax.grid(True, ls=":", lw=0.6, color="gray", alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=12.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.5)
    ax.set_title(title, fontsize=12.5, pad=7)
    ax.tick_params(labelsize=11)


def e1_panel(ax, dom, rows):
    rho = RHO[dom]
    rstar = RSTAR[dom]
    rs, fm, fs = seed_stats([r for r in rows if r["arm"] == "fixed"], ["r"], "q_measured")
    _, sm, ss = seed_stats([r for r in rows if r["arm"] == "adaptive"], ["r"], "q_measured")
    ax.errorbar(rs, fm, yerr=fs, fmt="o", color=BLUE, ms=5.5, mec="white", mew=0.6,
                elinewidth=1.0, capsize=2.5, label="Fixed (measured)", zorder=3)
    ax.errorbar(rs, sm, yerr=ss, fmt="s", color=ORANGE, ms=5, mec="white", mew=0.6,
                elinewidth=1.0, capsize=2.5, label="Strategic (measured)", zorder=3)
    rr = np.linspace(0, 0.95, 200)
    eta = 1 - rr
    X = 80 / 176
    ax.plot(rr, eta * X / (eta * X + rho), ls="--", color="black", lw=1.7,
            label="Fixed (model)", zorder=2)
    ax.plot(rr, [solve_brc_q(1 - r, 1.0, 2, rho) for r in rr], ls="-.", color="black",
            lw=1.7, label="Strategic (model)", zorder=2)
    ax.axhline(QBAR, ls=":", color="gray", lw=1.2, zorder=1)
    ax.text(0.985, QBAR + 0.02, r"$\bar q=0.3$", ha="right", va="bottom",
            fontsize=10, color="dimgray")
    ax.axvline(rstar, ls=":", color="gray", lw=1.2, zorder=1)
    ax.text(rstar + 0.02, 0.03, r"$r^\star$", ha="left", va="bottom",
            fontsize=11, color="dimgray")
    fixed_err = np.mean([float(r["abs_err"]) for r in rows if r["arm"] == "fixed"])
    strat_err = np.mean([float(r["abs_err"]) for r in rows if r["arm"] == "adaptive"])
    ax.text(0.972, 0.966, f"MAE fixed $= {fixed_err:.3f}$\nMAE strat. $= {strat_err:.3f}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=9.5, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black",
                      lw=0.9))
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    base_axis(ax, "Filter strength $r$", "Exposure $q$" if dom == "camera" else "",
              f"{dom} ($\\hat\\rho={rho:.3f}$)")
    leg = ax.legend(loc="lower left", frameon=True, framealpha=1.0, fancybox=False,
                    edgecolor="black", fontsize=7.6, handlelength=1.3,
                    borderpad=0.32, labelspacing=0.26, ncol=2, columnspacing=0.55,
                    handletextpad=0.4)
    leg.get_frame().set_linewidth(0.9)


def e1_figures():
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.4))
    for ax, dom in zip(axes, DOMAINS):
        rows = list(csv.DictReader(open(R / "e1" / f"e1_{dom}.csv")))
        e1_panel(ax, dom, rows)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIGD / "e1_filtering_panels.pdf", bbox_inches="tight")
    plt.close(fig)


def e3_panel(ax, dom, rows, col, legend_here=False):
    if col == "N":
        sel = [r for r in rows if r["arm"] == "equilibrium_w*"]
        xs, meas, stds = seed_stats(sel, ["N"], "q_measured")
        _, pred, _ = seed_stats(sel, ["N"], "q_pred")
        capped = {int(float(r["N"])): r.get("capped") == "1"
                  for r in sel if r.get("capped") is not None}
        ax.plot(xs, pred, ls="--", color="black", lw=1.7, zorder=2)
        for x, m, s in zip(xs, meas, stds):
            if capped.get(int(x)):
                ax.errorbar([x], [m], yerr=[s], fmt="o", ms=6, mec=BLUE, mew=1.3,
                            color="white", elinewidth=1.0, capsize=2.5, zorder=3)
            else:
                ax.errorbar([x], [m], yerr=[s], fmt="o", color=BLUE, ms=5.5,
                            mec="white", mew=0.6, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.plot([], [], ls="--", color="black", lw=1.7, label=r"Model $q^\star$")
        ax.errorbar([], [], fmt="o", color=BLUE, ms=5.5, mec="white", mew=0.6,
                    label="Measured")
        if any(capped.get(int(x), False) for x in xs):
            ax.errorbar([], [], fmt="o", ms=6, mec=BLUE, mew=1.3, color="white",
                        label="Capped (pool limit)")
        ax.set_xticks([1, 2, 4, 8])
        if legend_here:
            from matplotlib.lines import Line2D
            colr, mk = (BLUE, "o") if col == "N" else (ORANGE, "s")
            hs = [Line2D([], [], ls="--", color="black", lw=1.7, label=r"Model $q^\star$"),
                  Line2D([], [], ls="none", marker=mk, color=colr, ms=5.5,
                         mec="white", mew=0.6, label="Measured"),
                  Line2D([], [], ls="none", marker=mk, color="white", mec=colr,
                         mew=1.3, ms=6, label="Capped (pool limit)")]
            ax.legend(handles=hs, loc="lower right", frameon=True, framealpha=1.0,
                      fancybox=False, edgecolor="black", fontsize=8.5,
                      handlelength=1.8, borderpad=0.4, labelspacing=0.35)
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_xlabel("Providers $N$", fontsize=12.5)
        err = np.mean([abs(float(r["q_measured"]) - float(r["q_pred"]))
                       for r in sel if r.get("capped") != "1"])
        ax.text(0.035, 0.965, f"MAE $= {err:.3f}$", transform=ax.transAxes, va="top",
                ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="black", lw=0.9))
        base_axis(ax, "", "Exposure $q$", f"{dom}")
    else:
        sel = [r for r in rows if r["arm"].startswith("omega_")]
        agg, pred, capped = defaultdict(list), {}, {}
        for r in sel:
            w = float(r["arm"].split("_")[1])
            agg[w].append(float(r["q_measured"]))
            pred[w] = float(r["q_pred"])
            capped[w] = r.get("capped") == "1"
        ws = sorted(agg)
        meas = np.array([np.mean(agg[w]) for w in ws])
        stds = np.array([np.std(agg[w], ddof=1) for w in ws])
        pv = np.array([pred[w] for w in ws])
        ax.plot(ws, pv, ls="--", color="black", lw=1.7, zorder=2)
        for w, m, s in zip(ws, meas, stds):
            if capped.get(w):
                ax.errorbar([w], [m], yerr=[s], fmt="s", ms=5.5, mec=ORANGE, mew=1.3,
                            color="white", elinewidth=1.0, capsize=2.5, zorder=3)
            else:
                ax.errorbar([w], [m], yerr=[s], fmt="s", color=ORANGE, ms=5,
                            mec="white", mew=0.6, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.set_xticks(ws)
        ax.set_xticklabels([str(w) for w in ws])
        ax.set_xlabel(r"Benefit ratio $\omega$", fontsize=12.5)
        if legend_here:
            from matplotlib.lines import Line2D
            hs = [Line2D([], [], ls="--", color="black", lw=1.7, label=r"Model $q^\star$"),
                  Line2D([], [], ls="none", marker="s", color=ORANGE, ms=5,
                         mec="white", mew=0.6, label="Measured"),
                  Line2D([], [], ls="none", marker="s", color="white", mec=ORANGE,
                         mew=1.3, ms=5.5, label="Capped (pool limit)")]
            ax.legend(handles=hs, loc="lower right", frameon=True, framealpha=1.0,
                      fancybox=False, edgecolor="black", fontsize=8.5,
                      handlelength=1.8, borderpad=0.4, labelspacing=0.35)
        base_axis(ax, "", "", "")


def e3_figures():
    for col, fname in [("N", "e3_nsweep_column.pdf"), ("omega", "e3_omega_column.pdf")]:
        fig, axes = plt.subplots(3, 1, figsize=(3.15, 6.9))
        for i, (ax, dom) in enumerate(zip(axes, DOMAINS)):
            rows = list(csv.DictReader(open(R / "e3" / f"e3_{dom}.csv")))
            e3_panel(ax, dom, rows, col, legend_here=(i == 0))
        if axes[0].get_legend() is not None:
            axes[0].get_legend().get_frame().set_linewidth(0.9)
        fig.tight_layout(h_pad=1.5)
        fig.savefig(FIGD / fname, bbox_inches="tight")
        plt.close(fig)


def main():
    e1_figures()
    e3_figures()
    stats = {}
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
    with open(R / "paper_fig_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print("figures written")


if __name__ == "__main__":
    main()
