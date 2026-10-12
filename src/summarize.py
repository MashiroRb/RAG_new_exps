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
DOMAINS = ("camera", "ev", "shoes")


def mean_by(rows, key_cols, val_col):
    agg = {}
    for r in rows:
        k = tuple(r[c] for c in key_cols)
        agg.setdefault(k, []).append(float(r[val_col]))
    return {k[0]: float(np.mean(v)) for k, v in agg.items()}


def main():
    lines = ["# Experiment Summary (auto-generated)", ""]

    for dom in DOMAINS:
        grid_p = R / "e0" / f"e0_grid_{dom}.csv"
        fit_p = R / "e0" / f"e0_fit_{dom}.json"
        if not fit_p.exists():
            continue
        fit = json.load(open(fit_p))
        lines += [f"## E0 calibration ({dom})",
                  f"- rho_hat = {fit['rho_hat']}, R2 = {fit['r2']}, MAE = {fit['mae']}, "
                  f"RMSE = {fit['rmse']} (n={fit['n_points']})", ""]
        if grid_p.exists() and dom == "camera":
            e0 = list(csv.DictReader(open(grid_p)))
            pts = mean_by(e0, ["X"], "exposure")
            pairs = sorted((float(k), v) for k, v in pts.items())
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o", ms=4, label="measured (3-seed mean)")
            xr = np.linspace(0, max(p[0] for p in pairs), 100)
            ax.plot(xr, xr / (xr + fit["rho_hat"]), "-", lw=1.2, label=f"model q=X/(X+{fit['rho_hat']})")
            ax.axhline(0.3, ls=":", lw=1, color="gray")
            ax.text(0.02, 0.31, "qbar=0.3", fontsize=8, va="bottom")
            ax.set_xlabel("injection X (normalised)")
            ax.set_ylabel("exposure q")
            ax.set_title("E0: exposure formation law")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTD / "e0_fit.png", dpi=160)
            plt.close(fig)

    for dom in DOMAINS:
        e1_p = R / "e1" / f"e1_{dom}.csv"
        if not e1_p.exists():
            continue
        e1 = list(csv.DictReader(open(e1_p)))
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for arm, marker in [("fixed", "o"), ("adaptive", "s")]:
            pts = mean_by([r for r in e1 if r["arm"] == arm], ["r"], "q_measured")
            prd = mean_by([r for r in e1 if r["arm"] == arm], ["r"], "q_pred")
            rs = sorted(pts)
            ax.plot(rs, [pts[r] for r in rs], marker=marker, ms=5, lw=1.4, label=f"measured ({arm})")
            ax.plot(rs, [prd[r] for r in rs], ls="--", lw=1.1, label=f"model ({arm})")
        ax.axhline(0.3, ls=":", lw=1, color="gray")
        ax.set_xlabel("filtering recall r")
        ax.set_ylabel("exposure q")
        ax.set_title(f"E1 ({dom}): fixed vs adaptive injection")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTD / f"e1_qr_plane_{dom}.png", dpi=160)
        plt.close(fig)
        fixed_err = np.mean([float(r["abs_err"]) for r in e1 if r["arm"] == "fixed"])
        adapt_err = np.mean([float(r["abs_err"]) for r in e1 if r["arm"] == "adaptive"])
        gaps = []
        for r in sorted(set(float(x["r"]) for x in e1)):
            f_ = [float(x["q_measured"]) for x in e1 if x["arm"] == "fixed" and abs(float(x["r"]) - r) < 1e-9]
            a_ = [float(x["q_measured"]) for x in e1 if x["arm"] == "adaptive" and abs(float(x["r"]) - r) < 1e-9]
            if f_ and a_:
                gaps.append((r, float(np.mean(a_) - np.mean(f_))))
        lines += [f"## E1 filtering response ({dom})",
                  f"- mean |q-meas - model|: fixed={fixed_err:.4f}, adaptive={adapt_err:.4f}",
                  "- strategic offset q_adapt - q_fixed by r: " + ", ".join(f"r={r:.2f}:{g:+.3f}" for r, g in gaps), ""]

    try:
        s2 = json.load(open(R / "e2" / "e2_camera_summary.json"))
        lines += ["## E2 authentic-mass lever (camera)"]
        for arm in ("abs80", "scaled"):
            if arm in s2:
                lines.append(f"- {arm}: rho_implied by corpus fraction {s2[arm]['rho_implied_mean']}, "
                             f"ratio to full {s2[arm]['ratio_to_full']}, rho/f constancy {s2[arm]['rho_per_frac_mean']}")
        lines.append("")
    except FileNotFoundError:
        lines += ["## E2 authentic-mass lever (camera)", "- MISSING", ""]

    for dom in DOMAINS:
        e3_p = R / "e3" / f"e3_{dom}.csv"
        if not e3_p.exists():
            continue
        e3 = list(csv.DictReader(open(e3_p)))
        eqs = [r for r in e3 if r["arm"].startswith("equilibrium")]
        om = [r for r in e3 if r["arm"].startswith("omega_")]
        asym = [r for r in e3 if r["arm"].startswith("asymmetric")]
        lines += [f"## E3 comparative statics ({dom})"]
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
            for w in sorted(set(r["arm"].split("_")[1] for r in om)):
                sel = [r for r in om if r["arm"].split("_")[1] == w]
                cap = " (cap-binding)" if any(r.get("capped") == "1" for r in sel) else ""
                lines.append(f"- omega={w}: x*={np.mean([float(r['x_star_docs']) for r in sel]):.0f} "
                             f"q_meas={np.mean([float(r['q_measured']) for r in sel]):.3f} "
                             f"vs q0={np.mean([float(r['q_pred']) for r in sel]):.3f}{cap}")
        if asym:
            sh = np.mean([[float(r["share_lowcost"]), float(r["share_highcost"])] for r in asym], axis=0)
            lines.append(f"- asymmetric (cost 0.5 vs 2.0): measured exposure shares = {sh[0]:.2f} vs {sh[1]:.2f}")
        lines.append("")

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
