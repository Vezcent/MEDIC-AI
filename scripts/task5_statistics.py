"""
Task 5: Statistical Analysis (RQ2)
Regression model linking graph features (Task 4) to cognitive scores (MMSE).

Since real Alzheimer cohort data (OASIS-3/ADNI) requires account registration,
this script demonstrates the full statistical pipeline using simulated subjects.
Replace the simulated data with real participant data when available.

Pipeline:
  1. Load graph features from Task 4 (or simulate multi-subject data)
  2. Generate/load cognitive scores (MMSE) and QC parameters (SNR)
  3. Run OLS regression: MMSE ~ graph_features + QC_covariates
  4. Evaluate model fit (R², p-values, residual diagnostics)
  5. Generate publication-ready report
"""
import os
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -- File paths ----------------------------------------------------------------
DERIV_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "dataset", "derivatives")
FEATURES_PATH = os.path.join(DERIV_DIR, "graph_features.npz")
STATS_PATH    = os.path.join(DERIV_DIR, "task5_statistics.csv")
REPORT_PATH   = os.path.join(DERIV_DIR, "task5_report.png")


def simulate_cohort(n_subjects=60, seed=42):
    """
    Simulate a multi-subject Alzheimer cohort for demonstration.
    In production, replace this with real data from OASIS-3 or ADNI.

    Groups:
      - HC  (Healthy Control):   20 subjects, MMSE 27-30
      - MCI (Mild Cognitive):    20 subjects, MMSE 21-26
      - AD  (Alzheimer Disease): 20 subjects, MMSE 10-20

    Features per subject (derived from graph analysis):
      - degree_mean, clustering_avg, path_length, global_efficiency, small_worldness
      - snr (QC covariate from Task 2)
    """
    rng = np.random.RandomState(seed)
    n_per_group = n_subjects // 3

    groups = []
    data_rows = []

    for group, mmse_range, feat_scale in [
        ("HC",  (27, 30), 1.0),
        ("MCI", (21, 26), 0.75),
        ("AD",  (10, 20), 0.50),
    ]:
        for i in range(n_per_group):
            mmse = rng.uniform(*mmse_range)
            # Graph features degrade with disease progression
            degree      = rng.normal(15 * feat_scale, 2)
            clustering  = rng.normal(0.4 * feat_scale, 0.08)
            path_len    = rng.normal(2.5 / feat_scale, 0.3)
            g_eff       = rng.normal(0.35 * feat_scale, 0.05)
            small_world = rng.normal(2.0 * feat_scale, 0.4)
            snr         = rng.normal(8.0, 1.5)  # QC: independent of disease

            data_rows.append({
                "subject_id": f"sub-{len(data_rows)+1:03d}",
                "group":       group,
                "mmse":        round(mmse, 1),
                "degree_mean": round(max(degree, 0.1), 3),
                "clustering":  round(max(clustering, 0.01), 4),
                "path_length": round(max(path_len, 0.5), 4),
                "global_eff":  round(max(g_eff, 0.01), 4),
                "small_world": round(max(small_world, 0.1), 4),
                "snr":         round(snr, 2),
            })

    df = pd.DataFrame(data_rows)
    print(f"  Simulated cohort: {len(df)} subjects "
          f"({n_per_group} HC, {n_per_group} MCI, {n_per_group} AD)")
    return df


def run_regression(df):
    """
    OLS Regression: MMSE ~ degree + clustering + path_length +
                           global_eff + small_world + snr

    Tests whether graph features predict cognitive decline,
    controlling for image quality (SNR).
    """
    feature_cols = ["degree_mean", "clustering", "path_length",
                    "global_eff", "small_world", "snr"]

    X = df[feature_cols].astype(float)
    X = sm.add_constant(X)
    y = df["mmse"].astype(float)

    model  = sm.OLS(y, X)
    result = model.fit()

    return result, feature_cols


def run_group_comparison(df):
    """
    One-way ANOVA + post-hoc for each graph feature across groups.
    Tests: HC vs MCI vs AD differences in brain network topology.
    """
    feature_cols = ["degree_mean", "clustering", "path_length",
                    "global_eff", "small_world"]
    results = []

    for feat in feature_cols:
        groups = [df[df["group"] == g][feat].values for g in ["HC", "MCI", "AD"]]
        f_stat, p_val = sp_stats.f_oneway(*groups)
        results.append({
            "feature":  feat,
            "F_stat":   round(f_stat, 3),
            "p_value":  p_val,
            "sig":      "***" if p_val < 0.001 else
                        "**"  if p_val < 0.01  else
                        "*"   if p_val < 0.05  else "ns",
            "HC_mean":  round(groups[0].mean(), 4),
            "MCI_mean": round(groups[1].mean(), 4),
            "AD_mean":  round(groups[2].mean(), 4),
        })

    return pd.DataFrame(results)


def run_task5():
    """Main Task 5 pipeline."""
    print("=" * 60)
    print("  TASK 5: STATISTICAL ANALYSIS (RQ2)")
    print("  MMSE ~ Graph Features + QC Covariates")
    print("=" * 60)
    t_total = time.time()
    os.makedirs(DERIV_DIR, exist_ok=True)

    # =========================================================================
    # STEP 1: Load / simulate data
    # =========================================================================
    print("\n[1/4] Preparing cohort data...")

    # Check if real graph features exist
    if os.path.exists(FEATURES_PATH):
        feat = np.load(FEATURES_PATH)
        print(f"  Found Task 4 output: {list(feat.keys())}")
        print(f"  (Using simulated multi-subject cohort for regression demo)")

    df = simulate_cohort(n_subjects=60)
    df.to_csv(STATS_PATH, index=False)
    print(f"  Data saved: {os.path.basename(STATS_PATH)}")
    print(f"\n  Sample:\n{df.head(3).to_string(index=False)}")

    # =========================================================================
    # STEP 2: OLS Regression
    # =========================================================================
    print("\n[2/4] Running OLS regression: MMSE ~ graph features + SNR...")
    result, feature_cols = run_regression(df)
    print(f"\n{result.summary()}")

    # =========================================================================
    # STEP 3: Group comparison (ANOVA)
    # =========================================================================
    print("\n[3/4] Running group comparison (ANOVA: HC vs MCI vs AD)...")
    anova_df = run_group_comparison(df)
    print(f"\n{anova_df.to_string(index=False)}")

    # =========================================================================
    # STEP 4: Generate report
    # =========================================================================
    print("\n[4/4] Generating Task 5 report...")
    _generate_report(df, result, anova_df, feature_cols)

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  TASK 5 COMPLETE  ({total:.1f}s)")
    print(f"  Key findings:")
    print(f"    R-squared     : {result.rsquared:.4f}")
    print(f"    Adj. R-squared: {result.rsquared_adj:.4f}")
    print(f"    F-statistic   : {result.fvalue:.2f} (p={result.f_pvalue:.2e})")
    sig_features = [f for f in feature_cols
                    if result.pvalues[f] < 0.05]
    print(f"    Significant predictors (p<0.05): {sig_features or 'None'}")
    print(f"{'=' * 60}")

    return result, anova_df


def _generate_report(df, ols_result, anova_df, feature_cols):
    """Multi-panel publication-style report for Task 5."""
    fig = plt.figure(figsize=(20, 16), facecolor="#0e1117")
    fig.suptitle("Task 5 Report – Statistical Analysis (MEDIC-AI)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.4,
                           left=0.06, right=0.96,
                           top=0.92, bottom=0.05)

    group_colors = {"HC": "#00d4ff", "MCI": "#ffb347", "AD": "#ff6b6b"}

    # -- Panel 1: MMSE distribution by group -----------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for g, color in group_colors.items():
        vals = df[df["group"] == g]["mmse"]
        ax1.hist(vals, bins=8, alpha=0.6, color=color, label=g, edgecolor="none")
    ax1.set_title("MMSE by group", color="white", fontsize=10)
    ax1.set_xlabel("MMSE score", color="#aaa", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax1)

    # -- Panel 2: Degree vs MMSE scatter ---------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for g, color in group_colors.items():
        sub = df[df["group"] == g]
        ax2.scatter(sub["degree_mean"], sub["mmse"], c=color, s=30,
                    alpha=0.7, label=g, edgecolors="#333", linewidths=0.5)
    # Add regression line
    x_deg = df["degree_mean"].values
    slope, intercept, r, p, _ = sp_stats.linregress(x_deg, df["mmse"].values)
    x_line = np.linspace(x_deg.min(), x_deg.max(), 50)
    ax2.plot(x_line, slope * x_line + intercept, color="white",
             linestyle="--", linewidth=1, alpha=0.8)
    ax2.set_title(f"Degree vs MMSE  (r={r:.3f}, p={p:.1e})",
                  color="white", fontsize=9)
    ax2.set_xlabel("Degree (mean)", color="#aaa", fontsize=8)
    ax2.set_ylabel("MMSE", color="#aaa", fontsize=8)
    ax2.legend(fontsize=6, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax2)

    # -- Panel 3: Global efficiency vs MMSE ------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for g, color in group_colors.items():
        sub = df[df["group"] == g]
        ax3.scatter(sub["global_eff"], sub["mmse"], c=color, s=30,
                    alpha=0.7, label=g, edgecolors="#333", linewidths=0.5)
    x_eff = df["global_eff"].values
    slope, intercept, r, p, _ = sp_stats.linregress(x_eff, df["mmse"].values)
    x_line = np.linspace(x_eff.min(), x_eff.max(), 50)
    ax3.plot(x_line, slope * x_line + intercept, color="white",
             linestyle="--", linewidth=1, alpha=0.8)
    ax3.set_title(f"Global Eff vs MMSE  (r={r:.3f}, p={p:.1e})",
                  color="white", fontsize=9)
    ax3.set_xlabel("Global efficiency", color="#aaa", fontsize=8)
    ax3.set_ylabel("MMSE", color="#aaa", fontsize=8)
    _style_ax(ax3)

    # -- Panel 4: SNR vs MMSE (QC check) ---------------------------------------
    ax4 = fig.add_subplot(gs[0, 3])
    for g, color in group_colors.items():
        sub = df[df["group"] == g]
        ax4.scatter(sub["snr"], sub["mmse"], c=color, s=30,
                    alpha=0.7, label=g, edgecolors="#333", linewidths=0.5)
    x_snr = df["snr"].values
    slope, intercept, r, p, _ = sp_stats.linregress(x_snr, df["mmse"].values)
    ax4.set_title(f"SNR vs MMSE  (r={r:.3f}, p={p:.2f})",
                  color="white", fontsize=9)
    ax4.set_xlabel("SNR (QC)", color="#aaa", fontsize=8)
    ax4.set_ylabel("MMSE", color="#aaa", fontsize=8)
    _style_ax(ax4)

    # -- Panel 5-6: Regression coefficients ------------------------------------
    ax5 = fig.add_subplot(gs[1, 0:2])
    coefs   = ols_result.params[1:]   # skip const
    ci      = ols_result.conf_int().iloc[1:]
    p_vals  = ols_result.pvalues[1:]
    y_pos   = np.arange(len(coefs))
    colors  = ["#00ff88" if p < 0.05 else "#666666" for p in p_vals]
    xerr    = np.array([coefs.values - ci.iloc[:, 0].values,
                        ci.iloc[:, 1].values - coefs.values])
    ax5.barh(y_pos, coefs.values, xerr=xerr, color=colors,
             edgecolor="none", height=0.6, alpha=0.85)
    ax5.axvline(0, color="#ff6b6b", linestyle="--", linewidth=0.8)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(coefs.index, fontsize=8)
    ax5.set_title(f"OLS Coefficients (R²={ols_result.rsquared:.3f}, "
                  f"p={ols_result.f_pvalue:.1e})", color="white", fontsize=10)
    ax5.set_xlabel("Coefficient (green = p<0.05)", color="#aaa", fontsize=8)
    _style_ax(ax5)

    # -- Panel 6: Residual plot ------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    residuals = ols_result.resid
    fitted    = ols_result.fittedvalues
    ax6.scatter(fitted, residuals, c="#00d4ff", s=20, alpha=0.6,
                edgecolors="#333", linewidths=0.3)
    ax6.axhline(0, color="#ff6b6b", linestyle="--", linewidth=0.8)
    ax6.set_title("Residuals vs Fitted", color="white", fontsize=10)
    ax6.set_xlabel("Fitted MMSE", color="#aaa", fontsize=8)
    ax6.set_ylabel("Residual", color="#aaa", fontsize=8)
    _style_ax(ax6)

    # -- Panel 7: Q-Q plot ----------------------------------------------------
    ax7 = fig.add_subplot(gs[1, 3])
    sm.qqplot(residuals, line="45", ax=ax7, color="#00d4ff",
              markerfacecolor="#00d4ff", markeredgecolor="#333",
              markersize=4, alpha=0.6)
    ax7.set_title("Q-Q Plot (normality check)", color="white", fontsize=10)
    ax7.set_facecolor("#1e2530")
    ax7.tick_params(colors="#aaa", labelsize=7)
    for sp in ax7.spines.values():
        sp.set_color("#444")
    ax7.get_lines()[1].set_color("#ff6b6b")

    # -- Panel 8: ANOVA box plots ----------------------------------------------
    feature_names = anova_df["feature"].values
    n_feat = len(feature_names)
    for idx in range(min(n_feat, 4)):
        ax = fig.add_subplot(gs[2, idx])
        feat = feature_names[idx]
        row  = anova_df[anova_df["feature"] == feat].iloc[0]

        box_data = [df[df["group"] == g][feat].values for g in ["HC", "MCI", "AD"]]
        bp = ax.boxplot(box_data, labels=["HC", "MCI", "AD"],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=1.5))
        for patch, color in zip(bp["boxes"],
                                [group_colors["HC"], group_colors["MCI"],
                                 group_colors["AD"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("#aaa")

        sig_label = row["sig"]
        ax.set_title(f"{feat}\nF={row['F_stat']:.1f}  {sig_label}",
                     color="white", fontsize=9)
        _style_ax(ax)

    fig.savefig(REPORT_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved: {os.path.basename(REPORT_PATH)}")


def _style_ax(ax):
    ax.set_facecolor("#1e2530")
    ax.tick_params(colors="#aaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#444")


if __name__ == "__main__":
    result, anova = run_task5()
