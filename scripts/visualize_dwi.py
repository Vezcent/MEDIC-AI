"""
visualize_dwi.py – Visual QC inspection of preprocessed DTI data
Output: dataset/derivatives/qc_report.png (saved to file, no GUI required)
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")           # No display needed, save directly to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -- File paths ----------------------------------------------------------------
DERIV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "dataset", "derivatives")
PROC_PATH = os.path.join(DERIV_DIR, "dwi_preprocessed_fast.nii.gz")
OUT_PATH  = os.path.join(DERIV_DIR, "qc_report.png")


def snr(volume):
    mask = volume > np.percentile(volume, 30)
    sig  = np.mean(volume[mask])
    nse  = np.std(volume[~mask]) + 1e-8
    return sig / nse


def make_qc_report(proc_path=PROC_PATH, out_path=OUT_PATH):
    print(f"Loading: {proc_path}")
    img  = nib.load(proc_path)
    data = img.get_fdata().astype(np.float32)
    nx, ny, nz, nvol = data.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.suptitle("QC Report – DTI Preprocessed (MEDIC-AI)",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 5, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.05, right=0.97,
                           top=0.92, bottom=0.06)

    cmap = "inferno"
    label_kw = dict(color="#aaaaaa", fontsize=8)

    # -- Row 1: 3 orthogonal slices of b0 volume (vol 0) --------------------------
    vol0 = data[..., 0]
    vmax = np.percentile(vol0, 99)

    slices = [
        (vol0[cx, :, :],  f"Sagittal  (x={cx})"),
        (vol0[:, cy, :],  f"Coronal   (y={cy})"),
        (vol0[:, :, cz],  f"Axial     (z={cz})"),
    ]
    for col, (sl, title) in enumerate(slices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f"b0 – {title}", color="white", fontsize=9)
        ax.axis("off")

    # -- Row 1 col 3-4: SNR per volume --------------------------------------------
    print("Computing SNR for all volumes...")
    snr_vals = [snr(data[..., i]) for i in range(nvol)]

    ax_snr = fig.add_subplot(gs[0, 3:5])
    ax_snr.plot(snr_vals, color="#00d4ff", linewidth=1.2)
    ax_snr.axhline(np.mean(snr_vals), color="#ff6b6b", linestyle="--",
                   linewidth=1, label=f"Mean={np.mean(snr_vals):.2f}")
    ax_snr.set_title("SNR per DWI volume", color="white", fontsize=9)
    ax_snr.set_xlabel("Volume index (b-value)", **label_kw)
    ax_snr.set_ylabel("SNR", **label_kw)
    ax_snr.legend(fontsize=8, facecolor="#1e2530", labelcolor="white")
    ax_snr.set_facecolor("#1e2530")
    ax_snr.tick_params(colors="#aaaaaa", labelsize=7)
    for sp in ax_snr.spines.values():
        sp.set_color("#444444")

    # -- Row 2: 5 evenly-spaced DWI volumes ----------------------------------------
    vol_idxs = np.linspace(0, nvol - 1, 5, dtype=int)
    for col, vi in enumerate(vol_idxs):
        ax = fig.add_subplot(gs[1, col])
        sl  = data[:, :, cz, vi]
        ax.imshow(sl.T, origin="lower", cmap=cmap,
                  vmin=0, vmax=np.percentile(sl, 99) + 1)
        ax.set_title(f"Vol {vi}  (z={cz})", color="white", fontsize=8)
        ax.axis("off")

    # -- Row 3 col 0-1: Voxel intensity histogram ---------------------------------
    ax_hist = fig.add_subplot(gs[2, 0:2])
    flat = data[data > 0].ravel()
    ax_hist.hist(flat, bins=80, color="#00d4ff", alpha=0.75,
                 edgecolor="none", density=True)
    ax_hist.set_title("Voxel intensity distribution (>0)", color="white", fontsize=9)
    ax_hist.set_xlabel("Intensity", **label_kw)
    ax_hist.set_ylabel("Density", **label_kw)
    ax_hist.set_facecolor("#1e2530")
    ax_hist.tick_params(colors="#aaaaaa", labelsize=7)
    for sp in ax_hist.spines.values():
        sp.set_color("#444444")

    # -- Row 3 col 2-4: Summary statistics text panel ------------------------------
    ax_txt = fig.add_subplot(gs[2, 2:5])
    ax_txt.axis("off")
    stats = [
        ("File",          os.path.basename(proc_path)),
        ("Shape (x,y,z,vol)", str(data.shape)),
        ("Dtype",         str(data.dtype)),
        ("Min / Max",     f"{data.min():.2f}  /  {data.max():.2f}"),
        ("Mean +/- Std",  f"{data.mean():.2f} +/- {data.std():.2f}"),
        ("SNR mean",      f"{np.mean(snr_vals):.2f}"),
        ("SNR min / max", f"{min(snr_vals):.2f}  /  {max(snr_vals):.2f}"),
        ("Voxel < 0",     str((data < 0).sum())),
        ("NaN voxels",    str(np.isnan(data).sum())),
        ("Inf voxels",    str(np.isinf(data).sum())),
        ("Valid volumes", f"{sum(1 for v in snr_vals if v > 0)} / {nvol}"),
        ("File size",
         f"{os.path.getsize(proc_path)/1024**2:.1f} MB"),
    ]

    y0 = 0.95
    for key, val in stats:
        ok  = val not in ("0",) or key in ("Voxel < 0", "NaN voxels", "Inf voxels")
        color = "#00ff88" if ok else "#ff4444"
        ax_txt.text(0.02, y0, f"  {key:<22}: ", color="#aaaaaa",
                    fontsize=9, transform=ax_txt.transAxes, va="top",
                    fontfamily="monospace")
        ax_txt.text(0.45, y0, val, color=color,
                    fontsize=9, transform=ax_txt.transAxes, va="top",
                    fontfamily="monospace")
        y0 -= 0.082

    ax_txt.set_title("QC Summary Statistics", color="white", fontsize=9, loc="left")
    ax_txt.set_facecolor("#1e2530")

    # -- Save ----------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n=== QC report saved to: {out_path} ===")
    return out_path


if __name__ == "__main__":
    out = make_qc_report()
    # Auto-open the image file after saving (Windows)
    os.startfile(out)
