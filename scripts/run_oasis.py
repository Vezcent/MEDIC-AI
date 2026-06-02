"""
OASIS-1 Pipeline: Structural MRI Graph Analysis for Alzheimer's Disease
========================================================================
Adapted for OASIS-1 disc7 dataset (T1-weighted MPRAGE, not DTI).

Pipeline overview:
  1. Parse metadata: extract CDR, MMSE, nWBV, age, sex from .txt files
  2. Load processed T1 images (Analyze .hdr/.img format, atlas-registered T88)
  3. Use FSL segmentation labels to define ROIs (tissue-based parcellation)
  4. Build structural correlation network across subjects
  5. Extract graph features per subject (from individual morphometric patterns)
  6. Statistical analysis: Graph features vs CDR/MMSE

Input:  E:\\Medic\\MEDIC-AI\\oasis\\o1_using\\disc7\\
Output: E:\\Medic\\MEDIC-AI\\result\\o1\\
"""
import os
import re
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
DATA_DIR   = r"E:\Medic\MEDIC-AI\oasis\o1_using\disc7"
OUTPUT_DIR = r"E:\Medic\MEDIC-AI\result\o1"


# =============================================================================
# STEP 1: Parse OASIS metadata
# =============================================================================
def parse_subject_metadata(txt_path):
    """Parse an OASIS-1 subject .txt file for clinical/demographic info."""
    info = {
        "session_id": "", "age": None, "sex": "", "hand": "",
        "educ": None, "ses": None, "cdr": None, "mmse": None,
        "etiv": None, "asf": None, "nwbv": None,
    }
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SESSION ID:"):
                info["session_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("AGE:"):
                val = line.split(":", 1)[1].strip()
                info["age"] = int(val) if val else None
            elif line.startswith("M/F:"):
                info["sex"] = line.split(":", 1)[1].strip()
            elif line.startswith("HAND:"):
                info["hand"] = line.split(":", 1)[1].strip()
            elif line.startswith("EDUC:"):
                val = line.split(":", 1)[1].strip()
                info["educ"] = int(val) if val else None
            elif line.startswith("SES:"):
                val = line.split(":", 1)[1].strip()
                info["ses"] = int(val) if val else None
            elif line.startswith("CDR:"):
                val = line.split(":", 1)[1].strip()
                info["cdr"] = float(val) if val else None
            elif line.startswith("MMSE:"):
                val = line.split(":", 1)[1].strip()
                info["mmse"] = int(val) if val else None
            elif line.startswith("eTIV:"):
                val = line.split(":", 1)[1].strip()
                info["etiv"] = float(val) if val else None
            elif line.startswith("ASF:"):
                val = line.split(":", 1)[1].strip()
                info["asf"] = float(val) if val else None
            elif line.startswith("nWBV:"):
                val = line.split(":", 1)[1].strip()
                info["nwbv"] = float(val) if val else None
    return info


def load_all_metadata(data_dir=DATA_DIR):
    """Scan all subjects and collect metadata into a DataFrame."""
    rows = []
    for subj_dir in sorted(glob.glob(os.path.join(data_dir, "OAS1_*"))):
        sid = os.path.basename(subj_dir)
        txt_path = os.path.join(subj_dir, f"{sid}.txt")
        if not os.path.exists(txt_path):
            continue
        info = parse_subject_metadata(txt_path)
        info["subject_dir"] = subj_dir

        # Locate image files
        t88_dir = os.path.join(subj_dir, "PROCESSED", "MPRAGE", "T88_111")
        masked_files = glob.glob(os.path.join(t88_dir, "*_masked_gfc.img"))
        info["t88_masked_img"] = masked_files[0] if masked_files else None

        fseg_files = glob.glob(os.path.join(subj_dir, "FSL_SEG", "*_fseg.img"))
        info["fseg_img"] = fseg_files[0] if fseg_files else None

        rows.append(info)

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# STEP 2: Load and process T1 images
# =============================================================================
def load_analyze_image(img_path):
    """Load an Analyze 7.5 format image (.hdr/.img pair)."""
    hdr_path = img_path.replace(".img", ".hdr")
    img = nib.load(hdr_path)
    data = img.get_fdata()
    return data, img.affine


# =============================================================================
# STEP 3: Build per-subject ROI features from FSL segmentation
# =============================================================================
def extract_roi_features(t1_data, seg_data):
    """
    Extract regional intensity features from T1 image using FSL segmentation.
    FSL_SEG labels: 0=background, 1=CSF, 2=GM (gray matter), 3=WM (white matter)

    Strategy: Divide GM/WM into grid-based sub-regions and compute
    intensity statistics per region to build a feature vector.
    """
    features = {}

    # Global tissue volumes (normalized)
    total_brain = np.sum(seg_data > 0)
    if total_brain == 0:
        total_brain = 1

    features["vol_csf"]  = np.sum(seg_data == 1) / total_brain
    features["vol_gm"]   = np.sum(seg_data == 2) / total_brain
    features["vol_wm"]   = np.sum(seg_data == 3) / total_brain

    # GM intensity statistics
    gm_mask = seg_data == 2
    if gm_mask.any():
        gm_vals = t1_data[gm_mask]
        features["gm_mean"]   = np.mean(gm_vals)
        features["gm_std"]    = np.std(gm_vals)
        features["gm_median"] = np.median(gm_vals)
    else:
        features["gm_mean"] = features["gm_std"] = features["gm_median"] = 0

    # WM intensity statistics
    wm_mask = seg_data == 3
    if wm_mask.any():
        wm_vals = t1_data[wm_mask]
        features["wm_mean"]   = np.mean(wm_vals)
        features["wm_std"]    = np.std(wm_vals)
        features["wm_median"] = np.median(wm_vals)
    else:
        features["wm_mean"] = features["wm_std"] = features["wm_median"] = 0

    # Regional GM analysis: divide brain into grid sub-regions
    n_div = 4  # 4x4x4 = 64 regions
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
                if len(gm_in_roi) > 5:
                    region_means.append(np.mean(gm_in_roi))
                else:
                    region_means.append(0.0)

    features["regional_gm"] = np.array(region_means)
    return features


# =============================================================================
# STEP 4: Build structural correlation network
# =============================================================================
def build_correlation_network(all_regional_gm, threshold=0.3):
    """
    Build a structural covariance network from regional GM intensities.
    Each ROI is a node; edge weight = Pearson correlation of GM intensity
    across all subjects.

    all_regional_gm: array (n_subjects x n_regions)
    """
    n_regions = all_regional_gm.shape[1]
    corr_matrix = np.corrcoef(all_regional_gm.T)  # region x region
    np.fill_diagonal(corr_matrix, 0)

    # Threshold: keep only significant positive correlations
    adj_matrix = corr_matrix.copy()
    adj_matrix[adj_matrix < threshold] = 0

    # Build graph
    G = nx.Graph()
    for i in range(n_regions):
        G.add_node(i, label=f"ROI-{i+1}")
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    return G, corr_matrix, adj_matrix


# =============================================================================
# STEP 5: Graph feature extraction (per-subject via ego-network approach)
# =============================================================================
def compute_graph_features(G):
    """Compute 5 core graph metrics from a network."""
    features = {}

    # 1. Degree
    deg = dict(G.degree())
    degree_arr = np.array([deg[n] for n in sorted(deg)])
    features["degree_mean"] = degree_arr.mean()
    features["degree_max"]  = degree_arr.max()

    # 2. Clustering coefficient
    cc = nx.clustering(G, weight="weight")
    cc_arr = np.array([cc[n] for n in sorted(cc)])
    features["clustering_avg"] = cc_arr.mean()

    # 3. Path length (largest connected component)
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()
    G_sub = G.subgraph(largest).copy()
    if G_sub.number_of_nodes() >= 2:
        features["path_length"] = nx.average_shortest_path_length(G_sub)
    else:
        features["path_length"] = float('inf')

    # 4. Global efficiency
    features["global_efficiency"] = nx.global_efficiency(G)

    # 5. Small-worldness (simplified)
    if G_sub.number_of_nodes() >= 4 and G.number_of_edges() > 0:
        C_real = nx.average_clustering(G)
        L_real = features["path_length"]
        # Compare to random graph
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
# STEP 6: Statistical analysis
# =============================================================================
def run_statistics(df):
    """OLS regression: nWBV / graph features vs CDR."""
    print("\n  --- OLS Regression: CDR ~ nWBV + vol_gm + vol_wm + age ---")
    df_clean = df.dropna(subset=["cdr", "nwbv", "age", "vol_gm", "vol_wm"])
    if len(df_clean) < 5:
        print("  [WARNING] Not enough subjects with complete data.")
        return None

    X = df_clean[["nwbv", "vol_gm", "vol_wm", "age"]].astype(float)
    X = sm.add_constant(X)
    y = df_clean["cdr"].astype(float)

    model = sm.OLS(y, X)
    result = model.fit()
    print(f"\n{result.summary()}")
    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_report(df, G, corr_matrix, adj_matrix, graph_feats,
                    degree_arr, ols_result, output_dir):
    """Multi-panel QC report."""
    fig = plt.figure(figsize=(20, 16), facecolor="#0e1117")
    fig.suptitle("OASIS-1 Disc7 – Structural MRI Graph Analysis (MEDIC-AI)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # -- Panel 1: CDR distribution
    ax1 = fig.add_subplot(gs[0, 0])
    df_cdr = df.dropna(subset=["cdr"])
    if len(df_cdr) > 0:
        cdr_counts = df_cdr["cdr"].value_counts().sort_index()
        colors = ["#00d4ff" if c == 0 else "#ffb347" if c == 0.5
                  else "#ff6b6b" for c in cdr_counts.index]
        ax1.bar(cdr_counts.index.astype(str), cdr_counts.values,
                color=colors, edgecolor="none")
    ax1.set_title("CDR Distribution", color="white", fontsize=10)
    ax1.set_xlabel("CDR", color="#aaa", fontsize=8)
    ax1.set_ylabel("Count", color="#aaa", fontsize=8)
    _style_ax(ax1)

    # -- Panel 2: nWBV by CDR
    ax2 = fig.add_subplot(gs[0, 1])
    cdr_groups = {"CDR=0": 0.0, "CDR=0.5": 0.5, "CDR>=1": 1.0}
    for label, cdr_val in [("CDR=0", 0), ("CDR=0.5", 0.5)]:
        sub = df_cdr[df_cdr["cdr"] == cdr_val]
        if len(sub) > 0:
            color = "#00d4ff" if cdr_val == 0 else "#ffb347"
            ax2.scatter(sub["age"], sub["nwbv"], c=color, s=30,
                        alpha=0.7, label=label, edgecolors="#333", linewidths=0.5)
    sub_ad = df_cdr[df_cdr["cdr"] >= 1]
    if len(sub_ad) > 0:
        ax2.scatter(sub_ad["age"], sub_ad["nwbv"], c="#ff6b6b", s=30,
                    alpha=0.7, label="CDR>=1", edgecolors="#333", linewidths=0.5)
    ax2.set_title("nWBV vs Age (by CDR)", color="white", fontsize=10)
    ax2.set_xlabel("Age", color="#aaa", fontsize=8)
    ax2.set_ylabel("nWBV", color="#aaa", fontsize=8)
    ax2.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax2)

    # -- Panel 3: Correlation matrix
    ax3 = fig.add_subplot(gs[0, 2:4])
    im = ax3.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest")
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title("ROI Correlation Matrix (across subjects)", color="white",
                  fontsize=10)
    ax3.set_xlabel("ROI", color="#aaa", fontsize=8)
    ax3.set_ylabel("ROI", color="#aaa", fontsize=8)
    ax3.tick_params(colors="#aaa", labelsize=6)

    # -- Panel 4: Graph visualization
    ax4 = fig.add_subplot(gs[1, 0:2])
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        deg_vals = degree_arr.astype(float)
        node_sizes = deg_vals * 15 + 20
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.3 + 1.5 * w / max_w for w in weights]
        nx.draw_networkx_edges(G, pos, ax=ax4, alpha=0.3,
                               width=edge_widths, edge_color="#555")
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax4, node_size=node_sizes,
                                        node_color=deg_vals, cmap=plt.cm.plasma,
                                        edgecolors="#333", linewidths=0.5)
        plt.colorbar(nodes, ax=ax4, shrink=0.6, label="Degree")
    ax4.set_title("Structural Covariance Network", color="white", fontsize=10)
    ax4.set_facecolor("#0e1117")
    ax4.axis("off")

    # -- Panel 5: Summary statistics
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.axis("off")
    stats_text = [
        f"Total subjects: {len(df)}",
        f"With CDR data: {df['cdr'].notna().sum()}",
        f"With MMSE data: {df['mmse'].notna().sum()}",
        f"",
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        f"Degree (mean): {graph_feats['degree_mean']:.2f}",
        f"Clustering: {graph_feats['clustering_avg']:.4f}",
        f"Path length: {graph_feats['path_length']:.4f}",
        f"Global efficiency: {graph_feats['global_efficiency']:.4f}",
        f"Small-worldness: {graph_feats['small_worldness']:.4f}",
    ]
    if ols_result:
        stats_text += [
            f"",
            f"OLS R²: {ols_result.rsquared:.4f}",
            f"OLS F-stat: {ols_result.fvalue:.2f} (p={ols_result.f_pvalue:.2e})",
        ]
    for i, line in enumerate(stats_text):
        color = "#00d4ff" if line.startswith(("Graph", "OLS")) else "#00ff88"
        ax5.text(0.05, 0.95 - i * 0.07, line, color=color, fontsize=9,
                 transform=ax5.transAxes, va="top", fontfamily="monospace")
    ax5.set_title("Summary", color="white", fontsize=10, loc="left")
    ax5.set_facecolor("#1e2530")

    # -- Panel 6: Tissue volumes by CDR
    ax6 = fig.add_subplot(gs[2, 0])
    df_tissue = df.dropna(subset=["cdr", "vol_gm"])
    if len(df_tissue) > 0:
        for cdr_val, color, label in [(0, "#00d4ff", "CDR=0"),
                                       (0.5, "#ffb347", "CDR=0.5")]:
            sub = df_tissue[df_tissue["cdr"] == cdr_val]
            if len(sub) > 0:
                ax6.scatter(sub["vol_gm"], sub["vol_wm"], c=color, s=30,
                            alpha=0.7, label=label, edgecolors="#333")
        sub_ad = df_tissue[df_tissue["cdr"] >= 1]
        if len(sub_ad) > 0:
            ax6.scatter(sub_ad["vol_gm"], sub_ad["vol_wm"], c="#ff6b6b",
                        s=30, alpha=0.7, label="CDR>=1", edgecolors="#333")
    ax6.set_title("GM vs WM Volume (by CDR)", color="white", fontsize=10)
    ax6.set_xlabel("GM fraction", color="#aaa", fontsize=8)
    ax6.set_ylabel("WM fraction", color="#aaa", fontsize=8)
    ax6.legend(fontsize=6, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax6)

    # -- Panel 7: Degree distribution
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.bar(range(len(degree_arr)), degree_arr, color="#00d4ff",
            edgecolor="none", width=0.8)
    ax7.axhline(degree_arr.mean(), color="#ff6b6b", linestyle="--",
                label=f"Mean={degree_arr.mean():.1f}")
    ax7.set_title("Node Degree", color="white", fontsize=10)
    ax7.set_xlabel("ROI", color="#aaa", fontsize=8)
    ax7.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax7)

    # -- Panel 8: nWBV vs MMSE
    ax8 = fig.add_subplot(gs[2, 2:4])
    df_mmse = df.dropna(subset=["mmse", "nwbv"])
    if len(df_mmse) > 1:
        ax8.scatter(df_mmse["nwbv"], df_mmse["mmse"], c="#00d4ff", s=30,
                    alpha=0.7, edgecolors="#333", linewidths=0.5)
        slope, intercept, r, p, _ = sp_stats.linregress(
            df_mmse["nwbv"].values, df_mmse["mmse"].values)
        x_line = np.linspace(df_mmse["nwbv"].min(), df_mmse["nwbv"].max(), 50)
        ax8.plot(x_line, slope * x_line + intercept, color="white",
                 linestyle="--", linewidth=1)
        ax8.set_title(f"nWBV vs MMSE  (r={r:.3f}, p={p:.3f})",
                      color="white", fontsize=10)
    else:
        ax8.set_title("nWBV vs MMSE (insufficient data)", color="white")
    ax8.set_xlabel("nWBV", color="#aaa", fontsize=8)
    ax8.set_ylabel("MMSE", color="#aaa", fontsize=8)
    _style_ax(ax8)

    report_path = os.path.join(output_dir, "oasis_report.png")
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
def run_oasis_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    print("=" * 60)
    print("  OASIS-1 STRUCTURAL MRI GRAPH ANALYSIS")
    print("  Disc 7 – Alzheimer's Disease")
    print("=" * 60)
    t_total = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Parse metadata ────────────────────────────────────────────────
    print("\n[1/6] Loading subject metadata...")
    df = load_all_metadata(data_dir)
    print(f"  Total subjects: {len(df)}")
    print(f"  With CDR: {df['cdr'].notna().sum()}")
    print(f"  With MMSE: {df['mmse'].notna().sum()}")
    print(f"  CDR values: {sorted(df['cdr'].dropna().unique())}")
    print(f"  Age range: {df['age'].min()} – {df['age'].max()}")

    # ── Step 2: Load T1 images & FSL segmentation ────────────────────────────
    print("\n[2/6] Loading T1 images and FSL segmentation...")
    all_features = []
    all_regional_gm = []
    valid_indices = []

    for idx, row in df.iterrows():
        sid = row["session_id"]
        t1_path  = row["t88_masked_img"]
        seg_path = row["fseg_img"]

        if not t1_path or not seg_path:
            print(f"  [SKIP] {sid}: missing image files")
            continue
        if not os.path.exists(t1_path) or not os.path.exists(seg_path):
            print(f"  [SKIP] {sid}: files not found")
            continue

        try:
            t1_data, _ = load_analyze_image(t1_path)
            seg_data, _ = load_analyze_image(seg_path)

            # Ensure shapes match (seg may be different resolution)
            if t1_data.shape != seg_data.shape:
                # Resize seg to match T1
                zoom_factors = [t / s for t, s in
                                zip(t1_data.shape, seg_data.shape)]
                seg_data = ndimage.zoom(seg_data, zoom_factors, order=0)

            feats = extract_roi_features(t1_data, seg_data)
            all_features.append(feats)
            all_regional_gm.append(feats["regional_gm"])
            valid_indices.append(idx)
            print(f"  [OK] {sid}: GM={feats['vol_gm']:.3f}, "
                  f"WM={feats['vol_wm']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")

    if len(all_features) == 0:
        print("[FATAL] No subjects processed successfully.")
        return

    # Add tissue volumes to dataframe
    for i, idx in enumerate(valid_indices):
        df.loc[idx, "vol_gm"]  = all_features[i]["vol_gm"]
        df.loc[idx, "vol_wm"]  = all_features[i]["vol_wm"]
        df.loc[idx, "vol_csf"] = all_features[i]["vol_csf"]
        df.loc[idx, "gm_mean"] = all_features[i]["gm_mean"]
        df.loc[idx, "wm_mean"] = all_features[i]["wm_mean"]

    print(f"\n  Successfully processed: {len(valid_indices)} subjects")

    # ── Step 3: Build correlation network ─────────────────────────────────────
    print("\n[3/6] Building structural covariance network...")
    regional_matrix = np.array(all_regional_gm)
    G, corr_matrix, adj_matrix = build_correlation_network(
        regional_matrix, threshold=0.3)
    print(f"  Network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # ── Step 4: Graph features ────────────────────────────────────────────────
    print("\n[4/6] Computing graph features...")
    graph_feats, degree_arr, cc_arr = compute_graph_features(G)
    for k, v in graph_feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Step 5: Statistical analysis ──────────────────────────────────────────
    print("\n[5/6] Running statistical analysis...")
    ols_result = run_statistics(df)

    # ── Step 6: Save results & report ────────────────────────────────────────
    print("\n[6/6] Saving results...")

    # Save metadata + features
    csv_path = os.path.join(output_dir, "subjects_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Subjects CSV: {csv_path}")

    # Save connectome
    conn_path = os.path.join(output_dir, "connectome.npy")
    np.save(conn_path, adj_matrix)
    print(f"  Connectome: {conn_path}")

    # Save graph features
    feat_path = os.path.join(output_dir, "graph_features.npz")
    np.savez(feat_path, **{k: np.array([v]) for k, v in graph_feats.items()},
             degree=degree_arr, clustering=cc_arr)
    print(f"  Graph features: {feat_path}")

    # Generate report
    generate_report(df, G, corr_matrix, adj_matrix, graph_feats,
                    degree_arr, ols_result, output_dir)

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE  ({total:.1f}s)")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_oasis_pipeline()
