import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "RAG_new_exps" / "results"
OUTD = R / "summary"
OUTD.mkdir(parents=True, exist_ok=True)
SEEDS = [101, 202, 303]


def mean_by(rows, key_cols, val_col):
    agg = {}
    for r in rows:
        k = tuple(r[c] for c in key_cols)
        agg.setdefault(k, []).append(float(r[val_col]))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    lines = ["# Experiment Summary (auto-generated)", ""]

    e0 = list(csv.DictReader(open(R / "e0" / "e0_fit_camera.json" if False else R / "e0" / "e0_grid_camera.csv")))
    fit = json.load(open(R / "e0" / "e0_fit_camera.json"))
    lines += ["## E0 calibration (camera)", f"- rho_hat = {fit['rho_hat']}, R2 = {fit['r2']}, MAE = {fit['mae']}, RMSE = {fit['rmse']} (n={fit['n_points']})", ""]
    for dom in ("ev", "shoes"):
        pth = R / "e0" / f"e0_fit_{dom}.json"
        if pth.exists():
            fd = json.load(open(pth))
            lines += [f"## E0 calibration ({dom})", f"- rho_hat = {fd['rho_hat']}, R2 = {fd['r2']}, MAE = {fd['mae']}, RMSE = {fd['rmse']} (n={fd['n_points']})", ""]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    pts = mean_by(e0, ["X"], "exposure")
    xs = sorted(float(k[0]) for k in pts)
    ax.plot(xs, [pts[(str(x),) if False else k] for k, x in zip(sorted(pts), xs)], "o", ms=4, label="measured (3-seed mean)")
    xr = np.linspace(0, max(xs), 100)
    ax.plot(xr, xr / (xr + fit["rho_hat"]), "-", lw=1.2, label=f"model q=X/(X+{fit['rho_hat']})")
    ax.axhline(0.3, ls=":", lw=1, color="gray")
    ax.text(0.02, 0.31, "qbar=0.3", fontsize=8, va="bottom")
    ax.set_xlabel("injection X (normalised)"); ax.set_ylabel("exposure q")
    ax.set_title("E0: exposure formation law"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTD / "e0_fit.png", dpi=160); plt.close(fig)

    try:
        e1 = list(csv.DictReader(open(R / "e1" / "e1_camera.csv")))
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for arm, marker in [("fixed", "o"), ("adaptive", "s")]:
            pts = mean_by([r for r in e1 if r["arm"] == arm], ["r"], "q_measured")
            prd = mean_by([r for r in e1 if r["arm"] == arm], ["r"], "q_pred")
            rs = sorted(float(t[0]) for t in pts)
            ax.plot(rs, [pts[(str(v),)] for v in rs], marker=marker, ms=5, lw=1.4, label=f"measured ({arm})")
            ax.plot(rs, [prd[(str(v),)] for v in rs], ls="--", lw=1.1, label=f"model ({arm})")
        ax.axhline(0.3, ls=":", lw=1, color="gray")
        ax.set_xlabel("filtering recall r"); ax.set_ylabel("exposure q")
        ax.set_title("E1: fixed vs adaptive injection (N=2)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUTD / "e1_qr_plane.png", dpi=160); plt.close(fig)
        fixed_err = np.mean([float(r["abs_err"]) for r in e1 if r["arm"] == "fixed"])
        adapt_err = np.mean([float(r["abs_err"]) for r in e1 if r["arm"] == "adaptive"])
        gaps = []
        for r in sorted(set(float(x["r"]) for x in e1)):
            f_ = [float(x["q_measured"]) for x in e1 if x["arm"] == "fixed" and abs(float(x["r"]) - r) < 1e-9]
            a_ = [float(x["q_measured"]) for x in e1 if x["arm"] == "adaptive" and abs(float(x["r"]) - r) < 1e-9]
            if f_ and a_:
                gaps.append((float(r), float(np.mean(a_) - np.mean(f_))))
        lines += ["## E1 filtering response", f"- mean |q-meas − model|: fixed={fixed_err:.4f}, adaptive={adapt_err:.4f}",
                  "- strategic offset q_adapt − q_fixed by r: " + ", ".join(f"r={r:.2f}:{g:+.3f}" for r, g in gaps), ""]
    except FileNotFoundError:
        lines += ["## E1 filtering response", "- MISSING", ""]

    try:
        s2 = json.load(open(R / "e2" / "e2_camera_summary.json"))
        lines += ["## E2 authentic-mass lever"]
        for arm in ("abs80", "scaled"):
            if arm in s2:
                lines.append(f"- {arm}: rho_implied by corpus fraction {s2[arm]['rho_implied_mean']}, "
                             f"ratio to full {s2[arm]['ratio_to_full']}, rho/f constancy {s2[arm]['rho_per_frac_mean']}")
        lines.append("")
    except FileNotFoundError:
        lines += ["## E2 authentic-mass lever", "- MISSING", ""]

    try:
        e3 = list(csv.DictReader(open(R / "e3" / "e3_camera.csv")))
        eqs = [r for r in e3 if r["arm"].startswith("equilibrium")]
        om = [r for r in e3 if r["arm"].startswith("omega_")]
        asym = [r for r in e3 if r["arm"].startswith("asymmetric")]
        if eqs:
            lines += ["## E3 N-comparative statics (equilibrium arms, r=0)"]
            for tag in sorted(set(r["arm"] for r in eqs)):
                byN = {}
                for r in (x for x in eqs if x["arm"] == tag):
                    byN.setdefault(int(r["N"]), []).append((float(r["x_star_docs"]), float(r["X_docs"]),
                                                            float(r["q_measured"]), float(r["q_pred"]),
                                                            int(r.get("capped", 0) or 0)))
                for N in sorted(byN):
                    v = np.mean(byN[N], axis=0)
                    cap = " (cap-binding)" if v[4] > 0.5 else ""
                    lines.append(f"- {tag} N={N}: x*/provider={v[0]:.1f} docs, total={v[1]:.1f} docs, "
                                 f"q_meas={v[2]:.3f} vs q0={v[3]:.3f}{cap}")
        if om:
            lines += ["- omega sweep (N=2, cap-clamped where x* exceeds pool), 3-seed means:"]
            for w in sorted(set(r["arm"].split("_")[1] for r in om)):
                sel = [r for r in om if r["arm"].split("_")[1] == w]
                cap = " (cap-binding)" if sum(r.get("capped") == "1" for r in sel) else ""
                lines.append(f"  - omega={w}: x*={float(np.mean([float(r['x_star_docs']) for r in sel])):.0f} "
                             f"q_meas={float(np.mean([float(r['q_measured']) for r in sel])):.3f} "
                             f"vs q0={float(np.mean([float(r['q_pred']) for r in sel])):.3f}{cap}")
        if asym:
            sh = np.mean([[float(r["share_lowcost"]), float(r["share_highcost"])] for r in asym], axis=0)
            lines += [f"- asymmetric (cost 0.5 vs 2.0): measured exposure shares = {sh[0]:.2f} vs {sh[1]:.2f} (low-cost provider takes the larger share)", ""]
        else:
            lines += [""]
    except FileNotFoundError:
        lines += ["## E3", "- MISSING", ""]

    try:
        k = list(csv.DictReader(open(R / "r2" / "kappa_table.csv")))
        lines += ["## E4 selection effect (kappa)"]
        for row in k:
            if row["detector"] == "max_sim" and row["nominal_recall"] in ("0.3", "0.5"):
                lines.append(f"- {row['domain']}/{row['doc_type']} r={row['nominal_recall']}: kappa={row['kappa']} "
                             f"(survival retrievable {row['p_survive_retrievable']} vs unretrievable {row['p_survive_unretrievable']})")
        lines.append("")
    except FileNotFoundError:
        lines += ["## E4 selection effect (kappa)", "- MISSING", ""]

    try:
        a = list(csv.reader(open(R / "r4" / "anchor_summary.csv")))
        tail = [r for r in a if r and r[0].startswith(("spearman", "monotonic"))]
        lines += ["## R4 generator anchor (from logged rerun)"] + [f"- {r[0]}: {r[1]}" for r in tail] + [""]
    except FileNotFoundError:
        pass

    (OUTD / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
