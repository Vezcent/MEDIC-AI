"""
OASIS-2 Pipeline: Longitudinal Structural MRI Graph Analysis
=============================================================
Adapted for OASIS-2 RAW dataset (T1-weighted MPRAGE, longitudinal).

Key difference from OASIS-1:
  - OASIS-2 is longitudinal (multiple visits per subject)
  - RAW data only (no FSL_SEG or PROCESSED folders)
  - Metadata from external CSV (oasis_longitudinal.csv)
  - We perform our own tissue segmentation using intensity thresholding

Pipeline:
  1. Load metadata CSV (CDR, MMSE, nWBV, Group, Visit)
  2. Load RAW T1 images (Analyze .hdr/.img format)
  3. Auto tissue segmentation (Otsu thresholding → CSF/GM/WM)
  4. Extract regional features from 64 grid ROIs
  5. Build structural covariance network across sessions
  6. Graph feature extraction (5 core metrics)
  7. Statistical analysis: CDR ~ nWBV + morphometric features + age
  8. Longitudinal analysis: track nWBV/CDR change over visits

Input:  E:\\Medic\\MEDIC-AI\\oasis\\o2_using\\OAS2_RAW_PART1\\
Output: E:\\Medic\\MEDIC-AI\\result\\o2\\
"""
import os
import time
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats, ndimage
import statsmodels.api as sm


# -- Configuration -------------------------------------------------------------
DATA_DIR    = r"E:\Medic\MEDIC-AI\oasis\o2_using\OAS2_RAW_PART1"
META_CSV    = r"E:\Medic\MEDIC-AI\oasis\o2_using\oasis_longitudinal.csv"
OUTPUT_DIR  = r"E:\Medic\MEDIC-AI\result\o2"


# =============================================================================
# STEP 1: Load metadata CSV
# =============================================================================
def load_metadata(csv_path=META_CSV):
    """Load OASIS-2 longitudinal demographics CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Rename for consistency
    df = df.rename(columns={
        "Subject ID": "subject_id",
        "MRI ID":     "mri_id",
        "Group":      "group",
        "Visit":      "visit",
        "MR Delay":   "mr_delay",
        "M/F":        "sex",
        "Age":        "age",
        "EDUC":       "educ",
        "SES":        "ses",
        "MMSE":       "mmse",
        "CDR":        "cdr",
        "eTIV":       "etiv",
        "nWBV":       "nwbv",
        "ASF":        "asf",
    })
    return df


# =============================================================================
# STEP 2: Load RAW T1 images
# =============================================================================
def load_raw_image(session_dir):
    """
    Load the first MPRAGE scan from a RAW session directory.
    OASIS-2 RAW files: mpr-1.nifti.hdr / mpr-1.nifti.img
    """
    raw_dir = os.path.join(session_dir, "RAW")
    if not os.path.isdir(raw_dir):
        return None, None

    # Prefer mpr-1 (first acquisition)
    hdr_files = sorted(glob.glob(os.path.join(raw_dir, "mpr-1*.hdr")))
    if not hdr_files:
        hdr_files = sorted(glob.glob(os.path.join(raw_dir, "*.hdr")))
    if not hdr_files:
        return None, None

    img = nib.load(hdr_files[0])
    data = img.get_fdata()
    return data, img.affine


# =============================================================================
# STEP 3: Automatic tissue segmentation (no FSL_SEG available)
# =============================================================================
def auto_segment(t1_data):
    """
    Simple 3-class tissue segmentation using Otsu-like thresholding.
    Returns label map: 0=background, 1=CSF, 2=GM, 3=WM

    Strategy:
      - Mask out background (< 5th percentile) using a fast stride downsample
      - Split brain voxels into 3 classes by intensity percentiles using a stride sample
      - Low intensity → CSF, Medium → GM, High → WM
    """
    nonzero_mask = t1_data > 0
    if not nonzero_mask.any():
        return np.zeros_like(t1_data, dtype=np.int8)

    nonzero_vals = t1_data[nonzero_mask]
    sample_size = 50000
    step = max(1, len(nonzero_vals) // sample_size)
    bg_sample = nonzero_vals[::step][:sample_size]
    bg_thresh = np.percentile(bg_sample, 5)

    brain_mask = t1_data > bg_thresh
    seg = np.zeros_like(t1_data, dtype=np.int8)
    brain_vals = t1_data[brain_mask]

    if len(brain_vals) == 0:
        return seg

    sample_size_brain = 50000
    step_brain = max(1, len(brain_vals) // sample_size_brain)
    brain_sample = brain_vals[::step_brain][:sample_size_brain]
    p33 = np.percentile(brain_sample, 33)
    p66 = np.percentile(brain_sample, 66)

    seg[brain_mask & (t1_data <= p33)] = 1     # CSF
    seg[brain_mask & (t1_data > p33) & (t1_data <= p66)] = 2  # GM
    seg[brain_mask & (t1_data > p66)] = 3      # WM

    return seg


# =============================================================================
# STEP 4: Extract ROI features
# =============================================================================
def extract_roi_features(t1_data, seg_data):
    """Extract regional intensity features using grid-based ROIs."""
    features = {}

    total_brain = np.sum(seg_data > 0)
    if total_brain == 0:
        total_brain = 1

    features["vol_csf"] = np.sum(seg_data == 1) / total_brain
    features["vol_gm"]  = np.sum(seg_data == 2) / total_brain
    features["vol_wm"]  = np.sum(seg_data == 3) / total_brain

    # GM intensity stats
    gm_mask = seg_data == 2
    if gm_mask.any():
        gm_vals = t1_data[gm_mask]
        features["gm_mean"] = np.mean(gm_vals)
        features["gm_std"]  = np.std(gm_vals)
    else:
        features["gm_mean"] = features["gm_std"] = 0

    # WM intensity stats
    wm_mask = seg_data == 3
    if wm_mask.any():
        wm_vals = t1_data[wm_mask]
        features["wm_mean"] = np.mean(wm_vals)
        features["wm_std"]  = np.std(wm_vals)
    else:
        features["wm_mean"] = features["wm_std"] = 0

    # Regional GM: 4x4x4 = 64 ROIs
    n_div = 4
    shape = t1_data.shape
    sx = np.linspace(0, shape[0], n_div + 1, dtype=int)
    sy = np.linspace(0, shape[1], n_div + 1, dtype=int)
    sz = np.linspace(0, shape[2], n_div + 1, dtype=int)

    region_means = []
    for ix in range(n_div):
        for iy in range(n_div):
            for iz in range(n_div):
                roi = t1_data[sx[ix]:sx[ix+1], sy[iy]:sy[iy+1], sz[iz]:sz[iz+1]]
                seg_roi = seg_data[sx[ix]:sx[ix+1], sy[iy]:sy[iy+1], sz[iz]:sz[iz+1]]
                gm_in_roi = roi[seg_roi == 2]
                region_means.append(np.mean(gm_in_roi) if len(gm_in_roi) > 5
                                    else 0.0)

    # Normalize regional GM by global GM mean to remove scanner-scaling bias
    gm_mean = features["gm_mean"]
    region_means = np.array(region_means)
    if gm_mean > 0:
        region_means = region_means / gm_mean

    features["regional_gm"] = region_means
    return features


# =============================================================================
# STEP 5: Build structural covariance network
# =============================================================================
def build_correlation_network(all_regional_gm, threshold=0.3):
    """Build correlation network from regional GM across sessions."""
    n_regions = all_regional_gm.shape[1]
    corr_matrix = np.corrcoef(all_regional_gm.T)
    np.fill_diagonal(corr_matrix, 0)

    # Handle NaN from constant columns
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    adj_matrix = corr_matrix.copy()
    adj_matrix[adj_matrix < threshold] = 0

    G = nx.Graph()
    for i in range(n_regions):
        G.add_node(i, label=f"ROI-{i+1}")
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    return G, corr_matrix, adj_matrix


# =============================================================================
# STEP 6: Graph features
# =============================================================================
def compute_graph_features(G):
    """Compute 5 core graph metrics."""
    features = {}

    deg = dict(G.degree())
    degree_arr = np.array([deg[n] for n in sorted(deg)])
    features["degree_mean"] = degree_arr.mean()
    features["degree_max"]  = degree_arr.max()

    cc = nx.clustering(G, weight="weight")
    cc_arr = np.array([cc[n] for n in sorted(cc)])
    features["clustering_avg"] = cc_arr.mean()

    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    G_sub = G.subgraph(largest).copy()
    if G_sub.number_of_nodes() >= 2:
        features["path_length"] = nx.average_shortest_path_length(G_sub)
    else:
        features["path_length"] = float('inf')

    features["global_efficiency"] = nx.global_efficiency(G)

    # Small-worldness
    if G_sub.number_of_nodes() >= 4 and G.number_of_edges() > 0:
        C_real = nx.average_clustering(G)
        L_real = features["path_length"]
        rng = np.random.RandomState(42)
        C_rands, L_rands = [], []
        for _ in range(5):
            G_rand = nx.gnm_random_graph(G.number_of_nodes(),
                                          G.number_of_edges(),
                                          seed=rng.randint(1e6))
            C_rands.append(nx.average_clustering(G_rand))
            comps = list(nx.connected_components(G_rand))
            lcc = max(comps, key=len)
            sub = G_rand.subgraph(lcc)
            if sub.number_of_nodes() >= 2:
                L_rands.append(nx.average_shortest_path_length(sub))
        C_rand = np.mean(C_rands) if C_rands else 1e-8
        L_rand = np.mean(L_rands) if L_rands else 1e-8
        gamma = C_real / (C_rand + 1e-8)
        lmbda = L_real / (L_rand + 1e-8)
        features["small_worldness"] = gamma / (lmbda + 1e-8)
    else:
        features["small_worldness"] = float('nan')

    return features, degree_arr, cc_arr


# =============================================================================
# STEP 7: Statistical analysis
# =============================================================================
def run_statistics(df):
    """Multiple regression models on OASIS-2 data."""
    results = {}

    # Model 1: CDR ~ nWBV + vol_gm + vol_wm + age
    print("\n  --- Model 1: CDR ~ nWBV + vol_gm + vol_wm + age ---")
    df1 = df.dropna(subset=["cdr", "nwbv", "age", "vol_gm", "vol_wm"])
    if len(df1) >= 10:
        X = df1[["nwbv", "vol_gm", "vol_wm", "age"]].astype(float)
        X = sm.add_constant(X)
        y = df1["cdr"].astype(float)
        result1 = sm.OLS(y, X).fit()
        print(f"\n{result1.summary()}")
        results["cdr_model"] = result1
    else:
        print(f"  [SKIP] Only {len(df1)} valid rows")

    # Model 2: MMSE ~ nWBV + age + sex
    print("\n  --- Model 2: MMSE ~ nWBV + age + sex ---")
    df2 = df.dropna(subset=["mmse", "nwbv", "age"])
    if len(df2) >= 10:
        df2 = df2.copy()
        df2["sex_num"] = (df2["sex"] == "M").astype(int)
        X = df2[["nwbv", "age", "sex_num"]].astype(float)
        X = sm.add_constant(X)
        y = df2["mmse"].astype(float)
        result2 = sm.OLS(y, X).fit()
        print(f"\n{result2.summary()}")
        results["mmse_model"] = result2
    else:
        print(f"  [SKIP] Only {len(df2)} valid rows")

    return results


# =============================================================================
# REPORT
# =============================================================================
def generate_report(df, G, corr_matrix, adj_matrix, graph_feats,
                    degree_arr, ols_results, output_dir):
    """Multi-panel report for OASIS-2."""
    fig = plt.figure(figsize=(22, 18), facecolor="#0e1117")
    fig.suptitle("OASIS-2 Longitudinal – Structural MRI Graph Analysis (MEDIC-AI)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.4,
                           left=0.05, right=0.96, top=0.93, bottom=0.04)

    grp_colors = {"Nondemented": "#00d4ff", "Demented": "#ff6b6b",
                  "Converted": "#ffb347"}

    # -- Panel 1: Group distribution
    ax1 = fig.add_subplot(gs[0, 0])
    grp_counts = df["group"].value_counts()
    colors = [grp_colors.get(g, "#888") for g in grp_counts.index]
    ax1.bar(grp_counts.index, grp_counts.values, color=colors, edgecolor="none")
    ax1.set_title("Group Distribution", color="white", fontsize=10)
    ax1.set_ylabel("Sessions", color="#aaa", fontsize=8)
    ax1.tick_params(axis='x', rotation=15)
    _style_ax(ax1)

    # -- Panel 2: CDR distribution
    ax2 = fig.add_subplot(gs[0, 1])
    df_cdr = df.dropna(subset=["cdr"])
    if len(df_cdr) > 0:
        cdr_counts = df_cdr["cdr"].value_counts().sort_index()
        cdr_colors = ["#00d4ff" if c == 0 else "#ffb347" if c == 0.5
                      else "#ff6b6b" for c in cdr_counts.index]
        ax2.bar(cdr_counts.index.astype(str), cdr_counts.values,
                color=cdr_colors, edgecolor="none")
    ax2.set_title("CDR Distribution", color="white", fontsize=10)
    ax2.set_xlabel("CDR", color="#aaa", fontsize=8)
    _style_ax(ax2)

    # -- Panel 3: nWBV vs Age (by group)
    ax3 = fig.add_subplot(gs[0, 2])
    for grp, color in grp_colors.items():
        sub = df[df["group"] == grp]
        if len(sub) > 0:
            ax3.scatter(sub["age"], sub["nwbv"], c=color, s=15,
                        alpha=0.6, label=grp, edgecolors="#333", linewidths=0.3)
    ax3.set_title("nWBV vs Age (by Group)", color="white", fontsize=10)
    ax3.set_xlabel("Age", color="#aaa", fontsize=8)
    ax3.set_ylabel("nWBV", color="#aaa", fontsize=8)
    ax3.legend(fontsize=6, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax3)

    # -- Panel 4: MMSE vs nWBV
    ax4 = fig.add_subplot(gs[0, 3])
    df_mmse = df.dropna(subset=["mmse", "nwbv"])
    if len(df_mmse) > 1:
        for grp, color in grp_colors.items():
            sub = df_mmse[df_mmse["group"] == grp]
            if len(sub) > 0:
                ax4.scatter(sub["nwbv"], sub["mmse"], c=color, s=15,
                            alpha=0.6, label=grp, edgecolors="#333", linewidths=0.3)
        slope, intercept, r, p, _ = sp_stats.linregress(
            df_mmse["nwbv"].values, df_mmse["mmse"].values)
        x_line = np.linspace(df_mmse["nwbv"].min(), df_mmse["nwbv"].max(), 50)
        ax4.plot(x_line, slope * x_line + intercept, color="white",
                 linestyle="--", linewidth=1)
        ax4.set_title(f"nWBV vs MMSE (r={r:.3f}, p={p:.1e})",
                      color="white", fontsize=9)
    ax4.set_xlabel("nWBV", color="#aaa", fontsize=8)
    ax4.set_ylabel("MMSE", color="#aaa", fontsize=8)
    _style_ax(ax4)

    # -- Panel 5: Correlation matrix
    ax5 = fig.add_subplot(gs[1, 0:2])
    im = ax5.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest")
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_title("ROI Correlation Matrix", color="white", fontsize=10)
    ax5.tick_params(colors="#aaa", labelsize=6)

    # -- Panel 6: Graph visualization
    ax6 = fig.add_subplot(gs[1, 2:4])
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        deg_vals = degree_arr.astype(float)
        node_sizes = deg_vals * 10 + 20
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.2 + 1.0 * w / max_w for w in weights]
        nx.draw_networkx_edges(G, pos, ax=ax6, alpha=0.2,
                               width=edge_widths, edge_color="#555")
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax6, node_size=node_sizes,
                                        node_color=deg_vals, cmap=plt.cm.plasma,
                                        edgecolors="#333", linewidths=0.5)
        plt.colorbar(nodes, ax=ax6, shrink=0.6, label="Degree")
    ax6.set_title("Structural Covariance Network", color="white", fontsize=10)
    ax6.set_facecolor("#0e1117")
    ax6.axis("off")

    # -- Panel 7: Summary table
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.axis("off")
    lines = [
        f"Total sessions loaded: {len(df[df['vol_gm'].notna()])}",
        f"Unique subjects: {df['subject_id'].nunique()}",
        f"Groups: {df['group'].value_counts().to_dict()}",
        f"",
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        f"Degree (mean): {graph_feats['degree_mean']:.2f}",
        f"Clustering: {graph_feats['clustering_avg']:.4f}",
        f"Path length: {graph_feats['path_length']:.4f}",
        f"Global efficiency: {graph_feats['global_efficiency']:.4f}",
        f"Small-worldness: {graph_feats['small_worldness']:.4f}",
    ]
    cdr_model = ols_results.get("cdr_model")
    if cdr_model:
        lines += [
            f"",
            f"CDR Model R²: {cdr_model.rsquared:.4f}",
            f"CDR Model F: {cdr_model.fvalue:.2f} (p={cdr_model.f_pvalue:.2e})",
        ]
    mmse_model = ols_results.get("mmse_model")
    if mmse_model:
        lines += [
            f"MMSE Model R²: {mmse_model.rsquared:.4f}",
            f"MMSE Model F: {mmse_model.fvalue:.2f} (p={mmse_model.f_pvalue:.2e})",
        ]
    for i, line in enumerate(lines):
        color = "#00d4ff" if line.startswith(("Graph", "CDR", "MMSE")) else "#00ff88"
        ax7.text(0.02, 0.95 - i * 0.065, line, color=color, fontsize=8,
                 transform=ax7.transAxes, va="top", fontfamily="monospace")
    ax7.set_title("Summary", color="white", fontsize=10, loc="left")
    ax7.set_facecolor("#1e2530")

    # -- Panel 8: Longitudinal nWBV trajectories
    ax8 = fig.add_subplot(gs[2, 2:4])
    df_long = df.dropna(subset=["nwbv", "visit"])
    # Pick subjects with >= 3 visits
    subj_counts = df_long["subject_id"].value_counts()
    multi_visit = subj_counts[subj_counts >= 3].index[:15]
    for sid in multi_visit:
        sub = df_long[df_long["subject_id"] == sid].sort_values("visit")
        grp = sub["group"].iloc[0]
        color = grp_colors.get(grp, "#888")
        ax8.plot(sub["visit"], sub["nwbv"], "o-", color=color,
                 markersize=3, linewidth=1, alpha=0.7)
    ax8.set_title("Longitudinal nWBV Trajectories (≥3 visits)",
                  color="white", fontsize=10)
    ax8.set_xlabel("Visit", color="#aaa", fontsize=8)
    ax8.set_ylabel("nWBV", color="#aaa", fontsize=8)
    # Manual legend
    for grp, color in grp_colors.items():
        ax8.plot([], [], "o-", color=color, label=grp, markersize=3)
    ax8.legend(fontsize=6, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax8)

    # -- Panel 9-12: Box plots by group
    box_feats = [("vol_gm", "GM Volume"), ("vol_wm", "WM Volume"),
                 ("gm_mean", "GM Intensity"), ("wm_mean", "WM Intensity")]
    for idx, (feat, title) in enumerate(box_feats):
        ax = fig.add_subplot(gs[3, idx])
        df_feat = df.dropna(subset=[feat, "group"])
        if len(df_feat) > 0:
            box_data = []
            labels = []
            box_colors = []
            for grp in ["Nondemented", "Demented", "Converted"]:
                sub = df_feat[df_feat["group"] == grp][feat].values
                if len(sub) > 0:
                    box_data.append(sub)
                    labels.append(grp[:6])
                    box_colors.append(grp_colors[grp])
            if box_data:
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                                widths=0.5,
                                medianprops=dict(color="white", linewidth=1.5))
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for el in ["whiskers", "caps"]:
                    for line in bp[el]:
                        line.set_color("#aaa")
        ax.set_title(title, color="white", fontsize=9)
        _style_ax(ax)

    report_path = os.path.join(output_dir, "oasis2_report.png")
    fig.savefig(report_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved: {report_path}")


def _style_ax(ax):
    ax.set_facecolor("#1e2530")
    ax.tick_params(colors="#aaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#444")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_oasis2_pipeline(data_dir=DATA_DIR, meta_csv=META_CSV,
                        output_dir=OUTPUT_DIR, force_recompute=False):
    print("=" * 60)
    print("  OASIS-2 LONGITUDINAL STRUCTURAL MRI GRAPH ANALYSIS")
    print("=" * 60)
    t_total = time.time()
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "subjects_data.csv")
    npy_path = os.path.join(output_dir, "regional_gm.npy")

    # Check for precomputed data
    recompute = True
    if not force_recompute and os.path.exists(csv_path) and os.path.exists(npy_path):
        try:
            print("\nFound precomputed subjects_data.csv and regional_gm.npy. Loading...")
            df = pd.read_csv(csv_path)
            regional_matrix = np.load(npy_path)
            if len(df) == len(regional_matrix):
                print(f"  Successfully loaded {len(df)} precomputed sessions.")
                recompute = False
            else:
                print("  Precomputed sizes do not match. Recomputing...")
        except Exception as e:
            print(f"  Error loading precomputed files: {e}. Recomputing...")

    if recompute:
        # ── Step 1: Load metadata ─────────────────────────────────────────────
        print("\n[1/7] Loading metadata CSV...")
        df = load_metadata(meta_csv)
        print(f"  Total rows: {len(df)}")
        print(f"  Subjects: {df['subject_id'].nunique()}")
        print(f"  Groups: {df['group'].value_counts().to_dict()}")
        print(f"  Age range: {df['age'].min()} – {df['age'].max()}")

        # ── Step 2: Match sessions to image files ────────────────────────────────
        print("\n[2/7] Scanning for available image files...")
        available_dirs = {}
        for d in glob.glob(os.path.join(data_dir, "OAS2_*")):
            sid = os.path.basename(d)
            available_dirs[sid] = d

        matched = df["mri_id"].isin(available_dirs.keys())
        print(f"  Metadata sessions: {len(df)}")
        print(f"  Available on disk: {matched.sum()}")
        df = df[matched].copy()

        # ── Step 3: Load images and extract features ─────────────────────────────
        print(f"\n[3/7] Loading T1 images and computing features ({len(df)} sessions)...")
        all_regional_gm = []
        valid_indices = []
        processed = 0
        skipped = 0

        for idx, row in df.iterrows():
            mri_id = row["mri_id"]
            session_dir = available_dirs.get(mri_id)
            if not session_dir:
                skipped += 1
                continue

            try:
                t1_data, _ = load_raw_image(session_dir)
                if t1_data is None:
                    skipped += 1
                    continue

                seg_data = auto_segment(t1_data)
                feats = extract_roi_features(t1_data, seg_data)

                df.loc[idx, "vol_gm"]  = feats["vol_gm"]
                df.loc[idx, "vol_wm"]  = feats["vol_wm"]
                df.loc[idx, "vol_csf"] = feats["vol_csf"]
                df.loc[idx, "gm_mean"] = feats["gm_mean"]
                df.loc[idx, "wm_mean"] = feats["wm_mean"]

                all_regional_gm.append(feats["regional_gm"])
                valid_indices.append(idx)
                processed += 1

                if processed % 20 == 0:
                    print(f"    ... {processed} sessions processed")

            except Exception as e:
                print(f"  [ERROR] {mri_id}: {e}")
                skipped += 1

        print(f"  Processed: {processed}, Skipped: {skipped}")

        if processed == 0:
            print("[FATAL] No sessions processed successfully.")
            return

        regional_matrix = np.array(all_regional_gm)

        # Save metadata and regional matrices for fast caching
        df.to_csv(csv_path, index=False)
        np.save(npy_path, regional_matrix)
        print(f"  Saved precomputed features to {csv_path} and {npy_path}")

    # ── Step 4: Build correlation network ─────────────────────────────────────
    print("\n[4/7] Building structural covariance network...")
    # Increase threshold to 0.4 now that intensities are normalized
    G, corr_matrix, adj_matrix = build_correlation_network(
        regional_matrix, threshold=0.4)
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 5: Graph features ────────────────────────────────────────────────
    print("\n[5/7] Computing graph features...")
    graph_feats, degree_arr, cc_arr = compute_graph_features(G)
    for k, v in graph_feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Step 6: Statistical analysis ──────────────────────────────────────────
    print("\n[6/7] Running statistical analysis...")
    ols_results = run_statistics(df)

    # ── Step 7: Save results & report ────────────────────────────────────────
    print("\n[7/7] Saving results...")

    conn_path = os.path.join(output_dir, "connectome.npy")
    np.save(conn_path, adj_matrix)
    print(f"  Connectome: {conn_path}")

    feat_path = os.path.join(output_dir, "graph_features.npz")
    np.savez(feat_path, **{k: np.array([v]) for k, v in graph_feats.items()},
             degree=degree_arr, clustering=cc_arr)
    print(f"  Graph features: {feat_path}")

    generate_report(df, G, corr_matrix, adj_matrix, graph_feats,
                    degree_arr, ols_results, output_dir)

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  OASIS-2 PIPELINE COMPLETE  ({total:.1f}s)")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv or "-f" in sys.argv
    run_oasis2_pipeline(force_recompute=force)
