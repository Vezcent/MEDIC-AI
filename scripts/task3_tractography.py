"""
Task 3: Brain Graph Construction (Tractography & Parcellation)
Builds a structural connectome from preprocessed DTI data.

Pipeline:
  1. Load preprocessed DWI + gradient table
  2. Fit DTI model -> FA, MD maps (diffusion metrics)
  3. Fit CSD model -> fODF (fiber orientation distribution)
  4. Run deterministic tractography -> streamlines
  5. Parcellate brain into ROIs (grid-based for sample data)
  6. Build connectivity matrix (connectome)
  7. Save connectome + visualizations
"""
import os
import time
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                    auto_response_ssst)
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.data import default_sphere


# -- Configuration -------------------------------------------------------------
DERIV_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "dataset", "derivatives")
PROC_PATH   = os.path.join(DERIV_DIR, "dwi_preprocessed_fast.nii.gz")
CONN_PATH   = os.path.join(DERIV_DIR, "connectome.npy")
REPORT_PATH = os.path.join(DERIV_DIR, "task3_report.png")


def create_grid_parcellation(shape, n_regions_per_axis=4):
    """
    Create a simple grid-based parcellation for sample/demo data.
    Divides the 3D volume into cubic ROIs.
    In production, replace with a proper atlas (e.g., AAL, Desikan-Killiany).

    Returns: labels array (same spatial shape), int labels 1..N
    """
    labels = np.zeros(shape[:3], dtype=np.int32)
    nx, ny, nz = shape[:3]
    n = n_regions_per_axis

    sx = np.linspace(0, nx, n + 1, dtype=int)
    sy = np.linspace(0, ny, n + 1, dtype=int)
    sz = np.linspace(0, nz, n + 1, dtype=int)

    region_id = 1
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                labels[sx[ix]:sx[ix+1], sy[iy]:sy[iy+1], sz[iz]:sz[iz+1]] = region_id
                region_id += 1

    return labels


def build_connectome(streamlines, labels, affine):
    """
    Build a connectivity matrix from streamlines and ROI labels.
    For each streamline, identify the start and end ROIs and increment
    the corresponding entry in the matrix.
    """
    from dipy.tracking.utils import connectivity_matrix
    M = connectivity_matrix(streamlines, affine, labels,
                            return_mapping=False,
                            mapping_as_streamlines=False)
    # Remove self-connections (diagonal)
    np.fill_diagonal(M, 0)
    # Symmetrize
    M = M + M.T
    return M


def run_task3(proc_path=PROC_PATH, fast_mode=True):
    """
    Main Task 3 pipeline: DTI fitting, tractography, connectome construction.
    """
    print("=" * 60)
    print("  TASK 3: BRAIN GRAPH CONSTRUCTION")
    print("  (Tractography & Parcellation)")
    print("=" * 60)

    os.makedirs(DERIV_DIR, exist_ok=True)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n[1/6] Loading data...")
    t_total = time.time()

    if os.path.exists(proc_path):
        print(f"  Found preprocessed file: {os.path.basename(proc_path)}")
        img  = nib.load(proc_path)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine
        # Need gradient table from original data
        fetch_stanford_hardi()
        _, gtab = read_stanford_hardi()
    else:
        print("  No preprocessed file found. Loading raw Stanford HARDI...")
        fetch_stanford_hardi()
        img, gtab = read_stanford_hardi()
        data   = img.get_fdata().astype(np.float32)
        affine = img.affine

    # Crop for fast mode if data is full-size
    if fast_mode and data.shape[0] > 50:
        cx, cy, cz = [s // 2 for s in data.shape[:3]]
        r = 20
        data = data[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r, :]
        print(f"  [FAST MODE] Cropped to: {data.shape}")
    else:
        print(f"  Shape: {data.shape}")

    bvals = gtab.bvals
    bvecs = gtab.bvecs
    print(f"  Volumes: {data.shape[-1]}, b-values: {np.unique(bvals).astype(int)}")

    # =========================================================================
    # STEP 2: Create brain mask
    # =========================================================================
    print("\n[2/6] Creating brain mask...")
    from dipy.segment.mask import median_otsu
    b0_data = data[..., bvals < 50].mean(axis=-1)
    _, mask = median_otsu(b0_data, median_radius=2, numpass=1)
    print(f"  Mask voxels: {mask.sum()} / {mask.size} "
          f"({100*mask.sum()/mask.size:.1f}%)")

    # =========================================================================
    # STEP 3: Fit DTI model -> FA, MD
    # =========================================================================
    print("\n[3/6] Fitting DTI model (FA, MD maps)...")
    t0 = time.time()
    tensor_model = TensorModel(gtab)
    tensor_fit   = tensor_model.fit(data, mask=mask)
    FA = tensor_fit.fa
    MD = tensor_fit.md
    FA[np.isnan(FA)] = 0
    MD[np.isnan(MD)] = 0

    # Save FA map
    fa_path = os.path.join(DERIV_DIR, "fa_map.nii.gz")
    nib.save(nib.Nifti1Image(FA.astype(np.float32), affine), fa_path)
    print(f"  DTI fit done in {time.time()-t0:.1f}s")
    print(f"  FA range: [{FA[mask].min():.3f}, {FA[mask].max():.3f}], "
          f"mean={FA[mask].mean():.3f}")
    print(f"  MD range: [{MD[mask].min():.6f}, {MD[mask].max():.6f}]")
    print(f"  FA map saved: {os.path.basename(fa_path)}")

    # =========================================================================
    # STEP 4: Fit CSD model -> fiber ODF peaks for tractography
    # =========================================================================
    print("\n[4/6] Fitting CSD model (fiber orientation)...")
    t0 = time.time()
    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.5)
    csd_model   = ConstrainedSphericalDeconvModel(gtab, response)
    csd_peaks   = peaks_from_model(
        csd_model, data, default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask,
        return_sh=False,
        parallel=False       # single-thread for stability on Windows
    )
    print(f"  CSD fit done in {time.time()-t0:.1f}s")

    # =========================================================================
    # STEP 5: Deterministic tractography
    # =========================================================================
    print("\n[5/6] Running deterministic tractography...")
    t0 = time.time()

    # Stopping criterion: stop tracking when FA drops below 0.15
    stopping_criterion = ThresholdStoppingCriterion(FA, 0.15)

    # Seed points: every voxel with FA > 0.2
    seed_mask = FA > 0.2
    from dipy.tracking.utils import seeds_from_mask
    seeds = seeds_from_mask(seed_mask, affine, density=1)
    print(f"  Seed points: {len(seeds)}")

    # Track
    streamline_generator = LocalTracking(
        csd_peaks, stopping_criterion, seeds, affine,
        step_size=0.5,
        max_cross=1
    )
    streamlines = Streamlines(streamline_generator)

    # Filter: keep streamlines with reasonable length
    lengths = [len(s) for s in streamlines]
    min_len, max_len = 10, 500
    streamlines_filtered = Streamlines(
        [s for s, l in zip(streamlines, lengths) if min_len <= l <= max_len]
    )

    print(f"  Tracking done in {time.time()-t0:.1f}s")
    print(f"  Total streamlines: {len(streamlines)}")
    print(f"  After filtering ({min_len}-{max_len} pts): {len(streamlines_filtered)}")

    if len(streamlines_filtered) == 0:
        print("  [WARNING] No streamlines passed the filter. "
              "Relaxing length threshold...")
        streamlines_filtered = Streamlines(
            [s for s, l in zip(streamlines, lengths) if l >= 3]
        )
        print(f"  After relaxed filter (>=3 pts): {len(streamlines_filtered)}")

    # =========================================================================
    # STEP 6: Parcellation + Connectivity matrix
    # =========================================================================
    print("\n[6/6] Building connectivity matrix (connectome)...")
    t0 = time.time()

    # Create grid parcellation (replace with atlas for real data)
    n_regions = 4  # 4x4x4 = 64 ROIs
    labels = create_grid_parcellation(data.shape, n_regions_per_axis=n_regions)
    total_rois = labels.max()
    print(f"  Parcellation: {n_regions}x{n_regions}x{n_regions} grid = "
          f"{total_rois} ROIs")

    # Build connectivity matrix
    conn_matrix = build_connectome(streamlines_filtered, labels, affine)
    # Trim to actual ROI count (remove row/col 0 which is background)
    conn_matrix = conn_matrix[1:total_rois+1, 1:total_rois+1]

    # Save
    np.save(CONN_PATH, conn_matrix)
    print(f"  Connectome shape: {conn_matrix.shape}")
    print(f"  Non-zero connections: {(conn_matrix > 0).sum()}")
    print(f"  Max connection weight: {conn_matrix.max()}")
    print(f"  Saved: {os.path.basename(CONN_PATH)}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # =========================================================================
    # Generate report figure
    # =========================================================================
    print("\n[+] Generating Task 3 QC report...")
    _generate_report(FA, MD, mask, conn_matrix, streamlines_filtered,
                     labels, affine, data.shape)

    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  TASK 3 COMPLETE  ({total_time:.1f}s total)")
    print(f"  Outputs in: {DERIV_DIR}")
    print(f"    - fa_map.nii.gz")
    print(f"    - connectome.npy  ({conn_matrix.shape[0]} ROIs)")
    print(f"    - task3_report.png")
    print(f"{'=' * 60}")

    return conn_matrix


def _generate_report(FA, MD, mask, conn_matrix, streamlines, labels,
                     affine, data_shape):
    """Generate a multi-panel QC report for Task 3."""
    fig = plt.figure(figsize=(18, 12), facecolor="#0e1117")
    fig.suptitle("Task 3 Report – Tractography & Connectome (MEDIC-AI)",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.4, wspace=0.35,
                           left=0.05, right=0.97,
                           top=0.90, bottom=0.08)
    cz = FA.shape[2] // 2

    # -- Panel 1: FA map (axial) -----------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(FA[:, :, cz].T, origin="lower", cmap="hot", vmin=0, vmax=0.8)
    ax1.set_title("FA map (axial)", color="white", fontsize=9)
    ax1.axis("off")

    # -- Panel 2: MD map (axial) -----------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(MD[:, :, cz].T, origin="lower", cmap="inferno",
               vmin=0, vmax=np.percentile(MD[mask], 99))
    ax2.set_title("MD map (axial)", color="white", fontsize=9)
    ax2.axis("off")

    # -- Panel 3: FA histogram -------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    fa_vals = FA[mask].ravel()
    ax3.hist(fa_vals, bins=60, color="#ff6b6b", alpha=0.8, edgecolor="none",
             density=True)
    ax3.axvline(fa_vals.mean(), color="white", linestyle="--", linewidth=1,
                label=f"Mean={fa_vals.mean():.3f}")
    ax3.set_title("FA distribution", color="white", fontsize=9)
    ax3.set_xlabel("FA", color="#aaa", fontsize=8)
    ax3.legend(fontsize=7, facecolor="#1e2530", labelcolor="white")
    ax3.set_facecolor("#1e2530")
    ax3.tick_params(colors="#aaa", labelsize=7)
    for sp in ax3.spines.values():
        sp.set_color("#444")

    # -- Panel 4: Streamline count stats ----------------------------------------
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    lengths = [len(s) for s in streamlines]
    stats_text = [
        f"Total streamlines: {len(streamlines)}",
        f"Mean length: {np.mean(lengths):.1f} pts" if lengths else "N/A",
        f"Max length: {max(lengths)} pts" if lengths else "N/A",
        f"Min length: {min(lengths)} pts" if lengths else "N/A",
        f"",
        f"Connectome: {conn_matrix.shape[0]} x {conn_matrix.shape[1]}",
        f"Non-zero edges: {(conn_matrix > 0).sum()}",
        f"Max weight: {conn_matrix.max():.0f}",
        f"Mean weight: {conn_matrix[conn_matrix > 0].mean():.1f}" 
            if (conn_matrix > 0).any() else "N/A",
    ]
    for i, line in enumerate(stats_text):
        ax4.text(0.05, 0.92 - i * 0.10, line, color="#00d4ff",
                 fontsize=9, transform=ax4.transAxes, va="top",
                 fontfamily="monospace")
    ax4.set_title("Tractography statistics", color="white", fontsize=9,
                  loc="left")
    ax4.set_facecolor("#1e2530")

    # -- Panel 5-6: Connectivity matrix (heatmap) ------------------------------
    ax5 = fig.add_subplot(gs[1, 0:2])
    if conn_matrix.max() > 0:
        im = ax5.imshow(np.log1p(conn_matrix), cmap="magma",
                        interpolation="nearest")
        plt.colorbar(im, ax=ax5, shrink=0.8, label="log(1 + weight)")
    else:
        ax5.imshow(conn_matrix, cmap="magma")
    ax5.set_title("Structural Connectome (log scale)", color="white",
                  fontsize=9)
    ax5.set_xlabel("ROI", color="#aaa", fontsize=8)
    ax5.set_ylabel("ROI", color="#aaa", fontsize=8)
    ax5.tick_params(colors="#aaa", labelsize=6)

    # -- Panel 7-8: Node degree bar chart --------------------------------------
    ax6 = fig.add_subplot(gs[1, 2:4])
    degrees = (conn_matrix > 0).sum(axis=1)
    roi_ids = np.arange(1, len(degrees) + 1)
    colors  = plt.cm.viridis(degrees / (degrees.max() + 1e-8))
    ax6.bar(roi_ids, degrees, color=colors, edgecolor="none", width=0.8)
    ax6.set_title("Node degree per ROI", color="white", fontsize=9)
    ax6.set_xlabel("ROI index", color="#aaa", fontsize=8)
    ax6.set_ylabel("Degree (# connections)", color="#aaa", fontsize=8)
    ax6.set_facecolor("#1e2530")
    ax6.tick_params(colors="#aaa", labelsize=6)
    for sp in ax6.spines.values():
        sp.set_color("#444")

    fig.savefig(REPORT_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Report saved: {os.path.basename(REPORT_PATH)}")


if __name__ == "__main__":
    conn = run_task3(fast_mode=True)
