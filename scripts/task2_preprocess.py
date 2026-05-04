"""
Task 2: DTI Image Preprocessing
Optimized: image cropping, multi-threading (num_processes), fast_mode for quick testing
"""
import os
import time
import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.denoise.patch2self import patch2self
from dipy.denoise.gibbs import gibbs_removal


def calculate_snr(data, b0_index=0):
    """Calculate simple SNR on b0 image"""
    b0_data = data[..., b0_index]
    mask = b0_data > np.percentile(b0_data, 30)   # Signal region
    signal = np.mean(b0_data[mask])
    noise  = np.std(b0_data[~mask]) if np.any(~mask) else np.std(b0_data) + 1e-8
    return signal / noise if noise != 0 else float('inf')


def preprocess_dwi(data_path=None, bval_path=None,
                   fast_mode=True, num_processes=4):
    """
    fast_mode=True  -> crop brain to small central region + multi-thread -> ~30 seconds
    fast_mode=False -> process entire image (use with real Alzheimer data)
    num_processes   -> number of parallel CPU cores (default 4)
    """
    print("=== STARTING TASK 2: DTI IMAGE PREPROCESSING ===")
    if fast_mode:
        print("[FAST MODE] Running on small cropped region for quick pipeline validation.")

    # -- Load data -----------------------------------------------------------------
    if not data_path or not os.path.exists(data_path):
        print("[+] Loading standard DTI data (Stanford HARDI) from Dipy...")
        fetch_stanford_hardi()
        img, gtab = read_stanford_hardi()
        data   = img.get_fdata()
        affine = img.affine
        bvals  = gtab.bvals
        print(f"    Original shape: {data.shape}")
    else:
        print(f"[+] Loading real data from: {data_path}")
        img    = nib.load(data_path)
        data   = img.get_fdata()
        affine = img.affine
        bvals  = np.loadtxt(bval_path) if bval_path and os.path.exists(bval_path) \
                 else np.concatenate([[0], np.full(data.shape[-1] - 1, 1000)])
        print(f"    Original shape: {data.shape}")

    # -- Fast mode: crop central region --------------------------------------------
    if fast_mode:
        cx, cy, cz = [s // 2 for s in data.shape[:3]]
        r = 20   # radius of 20 voxels per axis -> 40x40x40 block
        data   = data[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r, :]
        print(f"    Cropped shape: {data.shape}  (fast mode)")

    # -- 1. Pre-processing QC ------------------------------------------------------
    snr_before = calculate_snr(data)
    print(f"\n[QC] SNR before processing: {snr_before:.2f}")

    # -- 2. Denoising (Patch2Self) -------------------------------------------------
    print(f"\n[1/3] Denoising (Patch2Self) – {num_processes} CPU threads...")
    t0 = time.time()
    # 'ridge' is significantly faster than 'ols' (default) - no matrix inversion needed
    data_denoised = patch2self(
        data, bvals,
        model='ridge',          # <- ~3-5x faster than OLS
        b0_threshold=50,
        shift_intensity=True,
        clip_negative_vals=False
    )
    print(f"      Done in {time.time()-t0:.1f}s")

    # -- 3. Gibbs Ringing Removal --------------------------------------------------
    print("\n[2/3] Gibbs Ringing Removal...")
    t0 = time.time()
    data_unringed = gibbs_removal(data_denoised, num_processes=num_processes)
    print(f"      Done in {time.time()-t0:.1f}s")

    # -- 4. Post-processing QC -----------------------------------------------------
    snr_after = calculate_snr(data_unringed)
    print(f"\n[QC] SNR after processing: {snr_after:.2f}")
    print(f"[QC] SNR improvement: +{snr_after - snr_before:.2f}")

    # -- 5. Save results -----------------------------------------------------------
    print("\n[3/3] Saving preprocessed image...")
    out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "dataset", "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    suffix   = "_fast" if fast_mode else ""
    out_path = os.path.join(out_dir, f"dwi_preprocessed{suffix}.nii.gz")
    nib.save(nib.Nifti1Image(data_unringed.astype(np.float32), affine), out_path)

    print(f"\n=== COMPLETE – saved to: {out_path} ===")
    return out_path


if __name__ == "__main__":
    import multiprocessing
    # Auto-detect CPU core count
    cores = min(multiprocessing.cpu_count(), 8)
    print(f"Detected {cores} CPU cores, using up to {cores} threads.")

    # -- Set fast_mode=False when using real Alzheimer data --
    preprocess_dwi(fast_mode=True, num_processes=cores)
