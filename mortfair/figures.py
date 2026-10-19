"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mortfair import plotting
from mortfair.features import GROUPS, LABELS
from mortfair.plotting import AXIS, GRID, INK, INK_2, METHOD_COLOR, MODEL_COLOR, MUTED, OBSERVED, PREDICTED, SERIES

# Takeaway titles, written after reading the results (kept here so a title fix never needs a recompute).
TITLES = {
    "data": "Property value, loan-to-value and DTI are 6-16 times more often unreported on denied applications",
    "hero": "Blind to race, ethnicity, sex and age, the model reproduces the Black-White gap in denial rates but "
            "under-predicts denials\nfor American Indian, Hispanic and Asian applicants",
    "models": "Gradient boosting beats the spline logistic baseline (ROC AUC 0.886-0.890 vs 0.862); "
              "the three boosters are within noise",
    "shap": "The lender has the largest average effect, but debt-to-income is what pushes applications towards denial",
    "reasons": "SHAP's top factor matches the lender's stated reason for 82% of DTI denials and 50% of collateral denials",
    "agreement": "SHAP and LIME rarely agree on a case's top feature: LIME puts a rare missing-value flag first every time",
    "fairness": "The model approves 5.3 points fewer Black than White applicants, yet approves about 99% of those the "
                "lenders approved in every group",
    "mitigation": "Group thresholds cut the race approval gap from 5.7 to 1.0 points for 0.2 points of accuracy; "
                  "reweighing and\nequal-opportunity thresholds do not help",
    "odds": "Reported application fields account for most of the Black-White denial gap, but not the American Indian, "
            "Hispanic\nor Asian gaps",
    "proxy": "The inputs encode race, ethnicity and age (AUC 0.75-0.78) without being given them; DTI is the one "
             "feature the model cannot lose",
}
ATTR_TITLE = {"race": "Race", "ethnicity": "Ethnicity", "sex": "Sex", "age": "Age"}


def _suptitle(fig: plt.Figure, text: str, gap_in: float = 0.3) -> None:
    """Left-aligned takeaway title placed ``gap_in`` inches above the top of the axes."""
    fig.suptitle(text, x=0.01, y=fig.subplotpars.top + gap_in / fig.get_figheight(), ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color=INK, wrap=True)


def _pct(ax: plt.Axes, axis: str = "x", digits: int = 0) -> None:
    fmt = plt.FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------------------------------------------- data
def fig_data(res: Path, path: Path) -> Path:
    """Population filter and missing-field shares by outcome."""
    steps = pd.read_csv(res / "population_filter.csv")
    miss = pd.read_csv(res / "missingness_by_outcome.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    yy = np.arange(len(steps))[::-1]
    a.barh(yy, steps.rows / 1e3, color=SERIES[0], height=0.6)
    for y_, (_, r) in zip(yy, steps.iterrows()):
        a.text(r.rows / 1e3 + 2, y_, f"{r.rows:,}  ({r.denial_rate:.1%} denied)", va="center", fontsize=8.5,
               color=INK_2)
    a.set_yticks(yy, steps.step, fontsize=8.5)
    a.set_xlim(0, steps.rows.max() / 1e3 * 1.55)
    a.set_xlabel("Applications (thousands)")
    a.set_title("Population filter (2025 Arizona home-purchase records)", loc="left")
    a.grid(axis="y", visible=False)
    names = {"property_value_k": "Property value", "cltv": "Combined LTV", "income_k": "Income", "dti": "Debt-to-income",
             "loan_term": "Loan term", "tract_income_pct": "Census tract"}
    m = miss.assign(name=miss.field.map(names)).iloc[::-1]
    yb = np.arange(len(m))
    b.barh(yb + 0.2, m.missing_approved, 0.4, color=SERIES[0], label="Approved applications")
    b.barh(yb - 0.2, m.missing_denied, 0.4, color=SERIES[7], label="Denied applications")
    b.set_yticks(yb, m.name)
    _pct(b, "x")
    b.set_xlabel("Share reported as 'NA'")
    b.set_title("Fields not reported, by outcome", loc="left")
    b.legend(loc="lower right")
    b.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.75, top=0.86)
    _suptitle(fig, TITLES["data"])
    plotting.save(fig, path)
    return path


# --------------------------------------------------------------------------------------------------------------- hero
def fig_observed_vs_predicted(res: Path, path: Path, min_n: int = 150) -> Path:
    """Observed denial rate per group (with CI) against the mean predicted denial probability of the unaware model."""
    audit = pd.read_csv(res / "fairness_audit.csv")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    for ax, attr in zip(axes, ["race", "ethnicity", "sex", "age"]):
        d = audit[(audit.attribute == attr) & (audit.n >= min_n)].copy()
        if attr == "age":
            order = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
            d = d.set_index("group").reindex([o for o in order if o in set(d.group)]).reset_index()
        else:
            d = d.sort_values("observed_denial_rate")
        yy = np.arange(len(d))
        ax.errorbar(d.observed_denial_rate, yy, xerr=[d.observed_denial_rate - d.observed_ci_low,
                                                      d.observed_ci_high - d.observed_denial_rate],
                    fmt="o", color=OBSERVED, ms=6, capsize=2, lw=1.2, label="Observed denial rate (95% CI)")
        ax.plot(d.mean_predicted, yy, "D", mfc="none", mec=PREDICTED, mew=1.6, ms=7,
                label="Predicted by the model (mean probability)")
        for y_, (_, r) in zip(yy, d.iterrows()):
            ax.plot([r.mean_predicted, r.observed_denial_rate], [y_, y_], color=GRID, lw=3, zorder=0)
        ax.set_yticks(yy, [f"{g}\n(n={n:,})" for g, n in zip(d.group, d.n)], fontsize=8)
        ax.set_title(ATTR_TITLE[attr], loc="left")
        ax.set_xlabel("Denial rate")
        _pct(ax, "x")
        ax.set_xlim(0, max(0.2, d.observed_ci_high.max() * 1.1))
        ax.grid(axis="y", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(x) for x in sorted(lab, key=lambda v: not v.startswith("Observed"))]
    axes[0].legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", bbox_to_anchor=(2.3, -0.2),
                   ncol=2)
    fig.subplots_adjust(wspace=0.75, top=0.84, bottom=0.2)
    _suptitle(fig, TITLES["hero"], gap_in=0.35)
    plotting.save(fig, path)
    return path


# ------------------------------------------------------------------------------------------------------------- models
def fig_models(res: Path, path: Path) -> Path:
    """Test-fold ROC AUC and AP with CIs, and calibration curves."""
    t = pd.read_csv(res / "model_results.csv")
    cal = pd.read_csv(res / "calibration_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    yy = np.arange(len(t))[::-1]
    for ax, m, lab in [(axes[0], "roc_auc", "ROC AUC"), (axes[1], "ap", "Average precision")]:
        for y_, (_, r) in zip(yy, t.iterrows()):
            c = MODEL_COLOR[r.model]
            ax.errorbar(r[m], y_, xerr=[[r[m] - r[f"{m}_ci_low"]], [r[f"{m}_ci_high"] - r[m]]], fmt="o", color=c,
                        capsize=3, ms=7)
            ax.text(r[f"{m}_ci_high"] + 0.004, y_, f"{r[m]:.3f}", va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(yy, t.model if ax is axes[0] else [""] * len(t))
        ax.set_xlabel(f"{lab} on the test fold (95% CI)")
        ax.set_title(lab, loc="left")
        ax.grid(axis="y", visible=False)
        lo, hi = t[f"{m}_ci_low"].min(), t[f"{m}_ci_high"].max()
        ax.set_xlim(lo - 0.02, hi + 0.04)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    for mname in t.model:
        d = cal[cal.model == mname]
        ax.plot(d.mean_predicted, d.observed, "-o", ms=3.5, lw=1.5, color=MODEL_COLOR[mname], label=mname)
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_yscale("symlog", linthresh=0.02)
    ticks = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks, [f"{v:.0%}" for v in ticks])
    ax.set_yticks(ticks, [f"{v:.0%}" for v in ticks])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted denial probability (decile)")
    ax.set_ylabel("Observed denial rate")
    ax.set_title("Calibration on the test fold", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.subplots_adjust(wspace=0.35, top=0.84)
    _suptitle(fig, TITLES["models"])
    plotting.save(fig, path)
    return path



# --------------------------------------------------------------------------------------------------------------- SHAP
def fig_shap(res: Path, path: Path, top: int = 14) -> Path:
    """Beeswarm of the strongest features and mean |SHAP| per feature group."""
    z = np.load(res / "shap_sample.npz", allow_pickle=False)
    sv, X, feats = z["shap"], z["X"], [str(f) for f in z["features"]]
    order = np.argsort(-np.abs(sv).mean(0))[:top][::-1]
    gi = pd.read_csv(res / "shap_group_importance.csv").sort_values("mean_abs_shap")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for row, j in enumerate(order):
        v, x = sv[:, j], X[:, j]
        if feats[j] in ("lender", "county", "aus", "credit_model"):
            col = np.full((len(v), 4), (*to_rgb(MUTED), 0.5))
        else:
            r = pd.Series(x).rank(pct=True).to_numpy()
            col = cmap(r)
            col[:, 3] = 0.55
        a.scatter(v, row + rng.uniform(-0.3, 0.3, len(v)), c=col, s=5, lw=0, rasterized=True)
    a.axvline(0, color=AXIS, lw=1)
    a.set_yticks(range(len(order)), [LABELS[feats[j]] for j in order], fontsize=8.5)
    a.set_xlabel("SHAP value (log-odds of denial)")
    a.set_title(f"Feature effects on {len(sv):,} test applications", loc="left")
    a.grid(axis="y", visible=False)
    handles = [Line2D([], [], marker="o", ls="", color=cmap(0.05), label="Low feature value"),
               Line2D([], [], marker="o", ls="", color=cmap(0.95), label="High feature value"),
               Line2D([], [], marker="o", ls="", color=MUTED, label="Categorical")]
    a.legend(handles=handles, loc="lower right", fontsize=8)
    b.barh(gi.group, gi.mean_abs_shap, color=SERIES[0], height=0.6)
    b.set_xlabel("Mean |sum of the group's SHAP values| (log-odds)")
    b.set_title("Importance by feature group", loc="left")
    b.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.55, top=0.88)
    _suptitle(fig, TITLES["shap"])
    plotting.save(fig, path)
    return path


def fig_reasons(res: Path, path: Path) -> Path:
    """Reported first denial reason against the feature group with the largest SHAP push towards denial."""
    ct = pd.read_csv(res / "reason_crosstab.csv")
    ct = ct[ct.n_denials >= 20].sort_values("n_denials", ascending=False)
    cols = list(GROUPS)
    share = ct[cols].div(ct.n_denials, axis=0).to_numpy()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.55 else INK_2)
    ax.set_xticks(range(len(cols)), [c.replace(" (", "\n(") for c in cols], fontsize=8.5, rotation=0)
    ax.set_yticks(range(len(ct)), [f"{r} (n={n:,})" for r, n in zip(ct.reported_reason, ct.n_denials)], fontsize=9)
    ax.set_xlabel("Feature group with the largest SHAP push towards denial (out-of-fold)")
    ax.set_ylabel("First denial reason reported by the lender")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Share of the reason's denials")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.subplots_adjust(top=0.88)
    _suptitle(fig, TITLES["reasons"])
    plotting.save(fig, path)
    return path


def fig_agreement(res: Path, path: Path) -> Path:
    """Top-k overlap of SHAP vs LIME and LIME vs LIME (another seed), and one denied case side by side."""
    ag = pd.read_csv(res / "explainer_agreement.csv")
    z = np.load(res / "local_attributions.npz", allow_pickle=False)
    feats = [str(f) for f in z["features"]]
    cases = pd.read_csv(res / "lime_cases.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    comps = [("SHAP vs LIME", SERIES[0]), ("SHAP vs LIME, missing-value flags excluded", SERIES[2]),
             ("LIME seed 0 vs seed 1", SERIES[1])]
    ks = sorted(ag.k.unique())
    rng = np.random.default_rng(1)
    w = 0.26
    for ci, (comp, col) in enumerate(comps):
        for ki, k in enumerate(ks):
            v = ag[(ag.comparison == comp) & (ag.k == k)].top_k_overlap.to_numpy()
            x = ki + (ci - 1) * w
            a.bar(x, v.mean(), w * 0.92, color=col, alpha=0.85, label=comp if ki == 0 else None)
            a.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v + rng.uniform(-0.015, 0.015, len(v)), s=7, color=INK,
                      alpha=0.3, lw=0)
            short = v.mean() < 0.1
            a.text(x, v.mean() + 0.02 if short else 0.02, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=8,
                   color=INK_2 if short else "white", fontweight="bold")
    a.set_xticks(range(len(ks)), [f"Top {k}" for k in ks])
    a.set_ylim(0, 1.4)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_ylabel("Share of top-k features in common")
    a.set_title(f"Agreement over {ag.case.nunique()} test applications", loc="left")
    a.legend(loc="upper left", fontsize=8)
    a.grid(axis="x", visible=False)
    # example: the denied, flagged case with the highest predicted probability
    den = cases[cases.stratum == "Denied, flagged"]
    i = int(den.sort_values("p_denial", ascending=False).case.iloc[0])
    s, lv = z["shap"][i], z["lime"][i]
    union = list(dict.fromkeys([*np.argsort(-np.abs(s))[:5], *np.argsort(-np.abs(lv))[:5]]))
    idx = np.array(sorted(union, key=lambda j: abs(s[j])))
    yy = np.arange(len(idx))
    b.barh(yy + 0.18, s[idx] / np.abs(s).sum(), 0.36, color=SERIES[0], label="SHAP (share of total |attribution|)")
    b.barh(yy - 0.18, lv[idx] / np.abs(lv).sum(), 0.36, color=SERIES[1], label="LIME (share of total |attribution|)")
    b.axvline(0, color=AXIS, lw=1)
    b.set_yticks(yy, [LABELS[feats[j]] for j in idx], fontsize=8.5)
    reason = den.set_index("case").loc[i, "reported_reason"]
    b.set_title(f"One denied application (p = {z['p'][i]:.2f}; lender's reason: {reason})", loc="left")
    b.set_xlabel("Signed attribution, normalised (positive = towards denial)")
    b.legend(loc="lower right", fontsize=8)
    b.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.45, top=0.86)
    _suptitle(fig, TITLES["agreement"])
    plotting.save(fig, path)
    return path


# ----------------------------------------------------------------------------------------------------------- fairness
def fig_fairness(res: Path, path: Path, min_n: int = 300) -> Path:
    """Model approval rate and true approval rate by race group, and within-group calibration curves."""
    audit = pd.read_csv(res / "fairness_audit.csv")
    cal = pd.read_csv(res / "group_calibration.csv")
    d = audit[(audit.attribute == "race") & (audit.n >= min_n)].sort_values("model_approval_rate")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    yy = np.arange(len(d))
    ref = d[d.group == "White"].iloc[0]
    for col, m, lab, off in [(SERIES[0], "model_approval_rate", "Approval rate (demographic parity)", 0.15),
                             (SERIES[2], "true_approval_rate", "Approval rate among applicants the lender approved "
                                                               "(equal opportunity)", -0.15)]:
        lo = d[m] - (ref[m] + d[f"{m}_diff_ci_low"])
        hi = (ref[m] + d[f"{m}_diff_ci_high"]) - d[m]
        a.errorbar(d[m], yy + off, xerr=[lo.clip(lower=0), hi.clip(lower=0)], fmt="o", color=col, capsize=2, label=lab)
    a.set_yticks(yy, [f"{g}{' (reference)' if g == 'White' else ''}\n(n={n:,})" for g, n in zip(d.group, d.n)],
                 fontsize=8.5)
    _pct(a, "x", 0)
    a.set_xlabel("Rate under the model's decisions (CI of the gap to White, placed at the group's value)")
    a.set_title("Race: model decisions at the accuracy-optimal threshold", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=8)
    a.grid(axis="y", visible=False)
    b.plot([0, 0.6], [0, 0.6], color=AXIS, ls="--", lw=1)
    groups = d.sort_values("n", ascending=False).group.tolist()
    for gi, g in enumerate(groups):
        c = cal[(cal.attribute == "race") & (cal.group == g)]
        b.plot(c.mean_predicted, c.observed, "-o", ms=3.5, lw=1.5, color=SERIES[gi % len(SERIES)], label=g)
    b.set_xscale("symlog", linthresh=0.02)
    b.set_yscale("symlog", linthresh=0.02)
    ticks = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    b.set_xticks(ticks, [f"{v:.0%}" for v in ticks])
    b.set_yticks(ticks, [f"{v:.0%}" for v in ticks])
    b.set_xlim(0, 0.75)
    b.set_ylim(0, 0.75)
    b.set_xlabel("Mean predicted denial probability (quintile)")
    b.set_ylabel("Observed denial rate")
    b.set_title("Within-group calibration (out-of-fold)", loc="left")
    b.legend(loc="upper left", fontsize=7.5)
    fig.subplots_adjust(wspace=0.4, top=0.86, bottom=0.25)
    _suptitle(fig, TITLES["fairness"])
    plotting.save(fig, path)
    return path


def fig_mitigation(res: Path, path: Path, min_n: int = 300) -> Path:
    """Accuracy against the race approval-rate gap per method, and approval rates by race group per method."""
    m = pd.read_csv(res / "mitigation_results.csv")
    g = pd.read_csv(res / "mitigation_by_group.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    for _, r in m.iterrows():
        c = METHOD_COLOR[r.method]
        a.errorbar(r.race_dp_difference, r.accuracy, xerr=[[r.race_dp_difference - r.dp_ci_low],
                                                           [r.dp_ci_high - r.race_dp_difference]],
                   yerr=[[r.accuracy - r.accuracy_ci_low], [r.accuracy_ci_high - r.accuracy]], fmt="o", ms=8, color=c,
                   capsize=3, label=r.method)
    _pct(a, "x", 1)
    _pct(a, "y", 1)
    a.set_xlabel("Largest gap in approval rate between race groups")
    a.set_ylabel("Accuracy against the lenders' decisions")
    a.set_title("Trade-off (out-of-fold, 95% CIs)", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=2)
    d = g[(g.attribute == "race") & (g.n >= min_n)]
    order = d[d.method == m.method.iloc[0]].sort_values("approval_rate").group.tolist()
    yy = np.arange(len(order))
    for k, meth in enumerate(m.method):
        dd = d[d.method == meth].set_index("group").reindex(order)
        b.plot(dd.approval_rate, yy + (k - 1.5) * 0.12, "o", color=METHOD_COLOR[meth], ms=6)
    b.set_yticks(yy, order, fontsize=8.5)
    _pct(b, "x", 1)
    b.set_xlabel("Model approval rate (colours as in the left panel)")
    b.set_title("Approval rate by race group and method", loc="left")
    b.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.4, top=0.86, bottom=0.27)
    _suptitle(fig, TITLES["mitigation"])
    plotting.save(fig, path)
    return path


def fig_odds(res: Path, path: Path) -> Path:
    """Forest plot of denial odds ratios: unadjusted, with application controls, and adding the lender."""
    o = pd.read_csv(res / "adjusted_odds_ratios.csv")
    keep = [("race", "Black"), ("race", "American Indian / Alaska Native"), ("race", "Asian"),
            ("race", "Not available"), ("ethnicity", "Hispanic or Latino"), ("sex", "Female"),
            ("sex", "Joint (male and female)"), ("age", "<25"), ("age", "65-74"), ("age", ">74")]
    ref = {"race": "White", "ethnicity": "not Hispanic", "sex": "male", "age": "35-44"}
    specs = ["Unadjusted", "Application controls", "Application controls + lender"]
    cols = [MUTED, SERIES[0], SERIES[1]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    rows = [(a, g) for a, g in keep if ((o.attribute == a) & (o.group == g)).any()]
    yy = np.arange(len(rows))[::-1]
    for si, (spec, col) in enumerate(zip(specs, cols)):
        for y_, (a, g) in zip(yy, rows):
            r = o[(o.attribute == a) & (o.group == g) & (o.specification == spec)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.odds_ratio, y_ + (1 - si) * 0.22, xerr=[[r.odds_ratio - r.ci_low], [r.ci_high - r.odds_ratio]],
                        fmt="o", color=col, capsize=2, ms=5, label=spec if y_ == yy[0] else None)
    ax.axvline(1, color=INK_2, lw=1)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(yy, [f"{g} (vs {ref[a]})" for a, g in rows], fontsize=9)
    ax.set_xlabel("Odds ratio of denial (95% CI, robust SE; log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.subplots_adjust(top=0.9)
    _suptitle(fig, TITLES["odds"])
    plotting.save(fig, path)
    return path


def fig_proxy_ablation(res: Path, path: Path) -> Path:
    """Feature-group ablation (test AUC change) and how well the inputs predict each protected attribute."""
    ab = pd.read_csv(res / "ablation.csv")
    px = pd.read_csv(res / "proxy_auc.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    d = ab[ab.feature_set != "All features"].sort_values("roc_auc_change", ascending=False)
    a.barh(d.feature_set, d.roc_auc_change, color=[SERIES[7] if v < 0 else SERIES[2] for v in d.roc_auc_change],
           height=0.6)
    a.axvline(0, color=INK_2, lw=1)
    a.set_xlabel(f"Change in test ROC AUC (all features: {ab.roc_auc.iloc[0]:.3f})")
    a.set_title("Feature-group ablation (tuned LightGBM, refit)", loc="left")
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", labelsize=8.5)
    yy = np.arange(len(px))[::-1]
    b.barh(yy + 0.18, px.auc_model_features, 0.36, color=SERIES[0], label="All model inputs (LightGBM)")
    b.barh(yy - 0.18, px.auc_tract_minority_share_either_direction, 0.36, color=SERIES[3],
           label="Tract minority share alone (excluded input;\nAUC in whichever direction is higher)")
    b.axvline(0.5, color=INK_2, lw=1, ls="--")
    for y_, (_, r) in zip(yy, px.iterrows()):
        b.text(r.auc_model_features + 0.01, y_ + 0.18, f"{r.auc_model_features:.2f}", va="center", fontsize=8)
        b.text(r.auc_tract_minority_share_either_direction + 0.01, y_ - 0.18,
               f"{r.auc_tract_minority_share_either_direction:.2f}", va="center", fontsize=8)
    b.set_yticks(yy, px.target, fontsize=8.5)
    b.set_xlim(0.4, 1)
    b.set_xlabel("ROC AUC for predicting the protected attribute (test fold)")
    b.set_title("Proxy check: group information in the inputs", loc="left")
    b.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), fontsize=8, ncol=2)
    b.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.6, top=0.86, bottom=0.22)
    _suptitle(fig, TITLES["proxy"])
    plotting.save(fig, path)
    return path


def make_all(results: Path, figures: Path) -> list[Path]:
    """Draw every figure from the result tables."""
    plotting.apply_style()
    return [fig_data(results, figures / "00_data.png"),
            fig_observed_vs_predicted(results, figures / "01_observed_vs_predicted.png"),
            fig_models(results, figures / "02_model_comparison.png"),
            fig_shap(results, figures / "03_shap_global.png"),
            fig_reasons(results, figures / "04_reason_codes.png"),
            fig_agreement(results, figures / "05_shap_vs_lime.png"),
            fig_fairness(results, figures / "06_fairness_audit.png"),
            fig_mitigation(results, figures / "07_mitigation.png"),
            fig_odds(results, figures / "08_odds_ratios.png"),
            fig_proxy_ablation(results, figures / "09_proxy_and_ablation.png")]
