"""
Task 4: Graph Feature Extraction
Computes 5 core network metrics from the structural connectome (Task 3 output).

Metrics:
  1. Degree           – number of connections per node
  2. Clustering coeff – tendency of neighbors to form clusters
  3. Path length      – average shortest path between all node pairs
  4. Global efficiency – inverse of average shortest path (whole-network)
  5. Small-worldness  – ratio comparing clustering/path length to random graphs
"""
import os
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -- File paths ----------------------------------------------------------------
DERIV_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "dataset", "derivatives")
CONN_PATH   = os.path.join(DERIV_DIR, "connectome.npy")
FEATURES_PATH = os.path.join(DERIV_DIR, "graph_features.npz")
REPORT_PATH = os.path.join(DERIV_DIR, "task4_report.png")


def load_connectome(path=CONN_PATH):
    """Load connectivity matrix and build a networkx graph."""
    print(f"  Loading connectome: {os.path.basename(path)}")
    M = np.load(path)
    # Build weighted undirected graph, skip zero-weight edges
    G = nx.Graph()
    n = M.shape[0]
    for i in range(n):
        G.add_node(i, label=f"ROI-{i+1}")
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > 0:
                G.add_edge(i, j, weight=M[i, j])
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return M, G


def compute_degree(G):
    """1. Degree: number of connections per node."""
    deg = dict(G.degree())
    return np.array([deg[n] for n in sorted(deg)])


def compute_clustering(G):
    """2. Clustering coefficient per node (weighted)."""
    cc = nx.clustering(G, weight="weight")
    return np.array([cc[n] for n in sorted(cc)])


def compute_path_length(G):
    """
    3. Characteristic path length (average shortest path).
    Uses inverse weight as distance (stronger connection = shorter path).
    Only computed on the largest connected component.
    """
    components = list(nx.connected_components(G))
    if len(components) > 1:
        largest_cc = max(components, key=len)
        G_sub = G.subgraph(largest_cc).copy()
        print(f"  [Note] Graph has {len(components)} components. "
              f"Using largest ({len(largest_cc)} nodes) for path length.")
    else:
        G_sub = G

    if G_sub.number_of_nodes() < 2:
        return float('inf')

    # Convert weights to distances (inverse weight)
    G_dist = G_sub.copy()
    for u, v, d in G_dist.edges(data=True):
        d["distance"] = 1.0 / (d["weight"] + 1e-8)

    return nx.average_shortest_path_length(G_dist, weight="distance")


def compute_global_efficiency(G):
    """
    4. Global efficiency: average of inverse shortest path lengths.
    E_global = (1 / N(N-1)) * sum(1 / d(i,j)) for all i != j
    """
    return nx.global_efficiency(G)


def compute_small_worldness(G, n_random=5, seed=42):
    """
    5. Small-worldness (sigma): S = (C/C_rand) / (L/L_rand)
    Compares clustering and path length to equivalent random graphs.
    """
    rng = np.random.RandomState(seed)

    # Get values for the real graph
    C_real = nx.average_clustering(G, weight="weight")

    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_sub = G.subgraph(largest_cc).copy()

    if G_sub.number_of_nodes() < 4:
        print("  [Warning] Graph too small for small-worldness calculation.")
        return float('nan'), C_real, 0, 0, 0

    L_real = nx.average_shortest_path_length(G_sub)

    # Generate random graphs with same degree sequence
    C_rands = []
    L_rands = []
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    for i in range(n_random):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges, seed=rng.randint(1e6))
        if G_rand.number_of_edges() == 0:
            continue
        C_rands.append(nx.average_clustering(G_rand))
        comps_rand = list(nx.connected_components(G_rand))
        lcc_rand = max(comps_rand, key=len)
        G_rand_sub = G_rand.subgraph(lcc_rand).copy()
        if G_rand_sub.number_of_nodes() >= 2:
            L_rands.append(nx.average_shortest_path_length(G_rand_sub))

    C_rand = np.mean(C_rands) if C_rands else 1e-8
    L_rand = np.mean(L_rands) if L_rands else 1e-8

    # Avoid division by zero
    gamma = C_real / (C_rand + 1e-8)    # clustering ratio
    lmbda = L_real / (L_rand + 1e-8)    # path length ratio
    sigma = gamma / (lmbda + 1e-8)      # small-worldness

    return sigma, C_real, L_real, C_rand, L_rand


def extract_features(conn_path=CONN_PATH):
    """Main pipeline: load connectome and compute all 5 features."""
    print("=" * 60)
    print("  TASK 4: GRAPH FEATURE EXTRACTION")
    print("=" * 60)
    t_total = time.time()

    # Load
    print("\n[1/5] Loading connectome...")
    M, G = load_connectome(conn_path)

    if G.number_of_edges() == 0:
        print("[ERROR] Connectome has no edges. Cannot compute features.")
        return None

    # 1. Degree
    print("\n[2/5] Computing degree...")
    t0 = time.time()
    degree = compute_degree(G)
    print(f"  Mean degree: {degree.mean():.2f}")
    print(f"  Max degree:  {degree.max()}")
    print(f"  Min degree:  {degree.min()}")
    print(f"  Done in {time.time()-t0:.3f}s")

    # 2. Clustering coefficient
    print("\n[3/5] Computing clustering coefficient...")
    t0 = time.time()
    clustering = compute_clustering(G)
    avg_clustering = np.mean(clustering[clustering > 0]) if (clustering > 0).any() \
                     else 0.0
    print(f"  Average clustering: {avg_clustering:.4f}")
    print(f"  Nodes with CC > 0: {(clustering > 0).sum()} / {len(clustering)}")
    print(f"  Done in {time.time()-t0:.3f}s")

    # 3. Characteristic path length
    print("\n[4/5] Computing characteristic path length...")
    t0 = time.time()
    path_length = compute_path_length(G)
    print(f"  Characteristic path length: {path_length:.4f}")
    print(f"  Done in {time.time()-t0:.3f}s")

    # 4. Global efficiency
    print("\n[5/5] Computing global efficiency & small-worldness...")
    t0 = time.time()
    g_eff = compute_global_efficiency(G)
    print(f"  Global efficiency: {g_eff:.4f}")

    # 5. Small-worldness
    sigma, C_real, L_real, C_rand, L_rand = compute_small_worldness(G)
    print(f"  Small-worldness (sigma): {sigma:.4f}")
    print(f"    C_real={C_real:.4f}, C_rand={C_rand:.4f}")
    print(f"    L_real={L_real:.4f}, L_rand={L_rand:.4f}")
    print(f"  Done in {time.time()-t0:.3f}s")

    # -- Save features ---------------------------------------------------------
    features = {
        "degree": degree,
        "clustering": clustering,
        "path_length": np.array([path_length]),
        "global_efficiency": np.array([g_eff]),
        "small_worldness": np.array([sigma]),
        "avg_clustering": np.array([avg_clustering]),
    }
    np.savez(FEATURES_PATH, **features)
    print(f"\n  Features saved: {os.path.basename(FEATURES_PATH)}")

    # -- Generate report -------------------------------------------------------
    print("\n[+] Generating Task 4 report...")
    _generate_report(M, G, degree, clustering, path_length, g_eff, sigma,
                     C_real, L_real, C_rand, L_rand)

    total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  TASK 4 COMPLETE  ({total:.1f}s)")
    print(f"  Summary:")
    print(f"    Degree (mean)          : {degree.mean():.2f}")
    print(f"    Clustering coeff (avg) : {avg_clustering:.4f}")
    print(f"    Path length            : {path_length:.4f}")
    print(f"    Global efficiency      : {g_eff:.4f}")
    print(f"    Small-worldness        : {sigma:.4f}")
    print(f"{'=' * 60}")

    return features


def _generate_report(M, G, degree, clustering, path_length, g_eff, sigma,
                     C_real, L_real, C_rand, L_rand):
    """Multi-panel QC report for Task 4."""
    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.suptitle("Task 4 Report – Graph Features (MEDIC-AI)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.96,
                           top=0.90, bottom=0.06)

    # -- Panel 1: Degree distribution -----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(degree)), degree, color="#00d4ff", edgecolor="none",
            width=0.8, alpha=0.85)
    ax1.axhline(degree.mean(), color="#ff6b6b", linestyle="--", linewidth=1,
                label=f"Mean={degree.mean():.2f}")
    ax1.set_title("1. Node Degree", color="white", fontsize=10)
    ax1.set_xlabel("ROI", color="#aaa", fontsize=8)
    ax1.set_ylabel("Degree", color="#aaa", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax1)

    # -- Panel 2: Clustering coefficient per node ------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    nonzero_cc = clustering[clustering > 0]
    if len(nonzero_cc) > 0:
        ax2.hist(nonzero_cc, bins=20, color="#ff6b6b", alpha=0.8,
                 edgecolor="none", density=True)
    ax2.axvline(np.mean(clustering), color="#00d4ff", linestyle="--",
                linewidth=1, label=f"Mean={np.mean(clustering):.4f}")
    ax2.set_title("2. Clustering Coefficient", color="white", fontsize=10)
    ax2.set_xlabel("CC value", color="#aaa", fontsize=8)
    ax2.set_ylabel("Density", color="#aaa", fontsize=8)
    ax2.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    _style_ax(ax2)

    # -- Panel 3: Summary table ------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    rows = [
        ("Metric", "Value"),
        ("─" * 25, "─" * 10),
        ("Degree (mean)", f"{degree.mean():.2f}"),
        ("Degree (max)", f"{degree.max()}"),
        ("Clustering (avg)", f"{np.mean(clustering):.4f}"),
        ("Path length", f"{path_length:.4f}"),
        ("Global efficiency", f"{g_eff:.4f}"),
        ("Small-worldness", f"{sigma:.4f}"),
        ("", ""),
        ("C_real / C_rand", f"{C_real:.4f} / {C_rand:.4f}"),
        ("L_real / L_rand", f"{L_real:.4f} / {L_rand:.4f}"),
        ("Nodes / Edges", f"{G.number_of_nodes()} / {G.number_of_edges()}"),
    ]
    for i, (k, v) in enumerate(rows):
        color_k = "#aaaaaa" if i > 0 else "#00d4ff"
        color_v = "#00ff88" if i > 1 else "#00d4ff"
        ax3.text(0.02, 0.95 - i * 0.075, k, color=color_k, fontsize=9,
                 transform=ax3.transAxes, va="top", fontfamily="monospace")
        ax3.text(0.65, 0.95 - i * 0.075, v, color=color_v, fontsize=9,
                 transform=ax3.transAxes, va="top", fontfamily="monospace")
    ax3.set_title("Feature Summary", color="white", fontsize=10, loc="left")
    ax3.set_facecolor("#1e2530")

    # -- Panel 4: Graph visualization (spring layout) --------------------------
    ax4 = fig.add_subplot(gs[1, 0:2])
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)
        # Node sizes proportional to degree
        node_sizes = degree * 30 + 20
        node_colors = degree.astype(float)
        # Edge widths proportional to weight
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.3 + 2.0 * w / max_w for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax4, alpha=0.3,
                               width=edge_widths, edge_color="#555555")
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax4,
                                        node_size=node_sizes,
                                        node_color=node_colors,
                                        cmap=plt.cm.plasma,
                                        edgecolors="#333333", linewidths=0.5)
        # Label only high-degree nodes
        high_deg = {n: f"{n+1}" for n in range(len(degree))
                    if degree[n] >= degree.mean()}
        nx.draw_networkx_labels(G, pos, high_deg, ax=ax4,
                                font_size=6, font_color="white")
        plt.colorbar(nodes, ax=ax4, shrink=0.6, label="Degree")
    ax4.set_title("Brain Network Graph (spring layout)", color="white",
                  fontsize=10)
    ax4.set_facecolor("#0e1117")
    ax4.axis("off")

    # -- Panel 5: Small-worldness comparison -----------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    categories = ["C_real", "C_rand", "L_real", "L_rand"]
    values     = [C_real, C_rand, L_real, L_rand]
    colors_bar = ["#00d4ff", "#555555", "#ff6b6b", "#555555"]
    bars = ax5.barh(categories, values, color=colors_bar, edgecolor="none",
                    height=0.6)
    ax5.set_title(f"Small-worldness  σ = {sigma:.2f}", color="white",
                  fontsize=10)
    ax5.set_xlabel("Value", color="#aaa", fontsize=8)
    for bar, val in zip(bars, values):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=8)
    _style_ax(ax5)

    # -- Panel 6: Degree vs Clustering scatter ---------------------------------
    ax6 = fig.add_subplot(gs[2, 0])
    mask = degree > 0
    if mask.sum() > 0:
        sc = ax6.scatter(degree[mask], clustering[mask], c=degree[mask],
                         cmap="plasma", s=40, alpha=0.8, edgecolors="#333")
        plt.colorbar(sc, ax=ax6, shrink=0.8, label="Degree")
    ax6.set_title("Degree vs Clustering", color="white", fontsize=10)
    ax6.set_xlabel("Degree", color="#aaa", fontsize=8)
    ax6.set_ylabel("Clustering coeff", color="#aaa", fontsize=8)
    _style_ax(ax6)

    # -- Panel 7: Weighted connectome (sorted by degree) -----------------------
    ax7 = fig.add_subplot(gs[2, 1:3])
    sort_idx = np.argsort(degree)[::-1]
    M_sorted = M[np.ix_(sort_idx, sort_idx)]
    if M_sorted.max() > 0:
        im = ax7.imshow(np.log1p(M_sorted), cmap="magma",
                        interpolation="nearest")
        plt.colorbar(im, ax=ax7, shrink=0.8, label="log(1 + weight)")
    ax7.set_title("Connectome (sorted by degree)", color="white", fontsize=10)
    ax7.set_xlabel("ROI (sorted)", color="#aaa", fontsize=8)
    ax7.set_ylabel("ROI (sorted)", color="#aaa", fontsize=8)
    ax7.tick_params(colors="#aaa", labelsize=6)

    fig.savefig(REPORT_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved: {os.path.basename(REPORT_PATH)}")


def _style_ax(ax):
    """Apply dark theme to axis."""
    ax.set_facecolor("#1e2530")
    ax.tick_params(colors="#aaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#444")


if __name__ == "__main__":
    features = extract_features()
