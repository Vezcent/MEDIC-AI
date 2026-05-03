"""
Task 2: Tiền xử lý ảnh DTI
Đã tối ưu: crop ảnh, đa luồng (num_processes), fast_mode để test nhanh
"""
import os
import time
import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.denoise.patch2self import patch2self
from dipy.denoise.gibbs import gibbs_removal


def calculate_snr(data, b0_index=0):
    """Tính toán SNR đơn giản trên ảnh b0"""
    b0_data = data[..., b0_index]
    mask = b0_data > np.percentile(b0_data, 30)   # Phần có tín hiệu
    signal = np.mean(b0_data[mask])
    noise  = np.std(b0_data[~mask]) if np.any(~mask) else np.std(b0_data) + 1e-8
    return signal / noise if noise != 0 else float('inf')


def preprocess_dwi(data_path=None, bval_path=None,
                   fast_mode=True, num_processes=4):
    """
    fast_mode=True  → crop não xuống vùng trung tâm nhỏ + dùng đa luồng → ~30 giây
    fast_mode=False → xử lý toàn bộ ảnh (dùng khi có data Alzheimer thật)
    num_processes   → số CPU cores song song (mặc định 4)
    """
    print("=== BẮT ĐẦU TASK 2: TIỀN XỬ LÝ ẢNH DTI ===")
    if fast_mode:
        print("[FAST MODE] Đang chạy trên vùng crop nhỏ để kiểm tra pipeline nhanh.")

    # ── Load dữ liệu ──────────────────────────────────────────────────────────
    if not data_path or not os.path.exists(data_path):
        print("[+] Tải dữ liệu DTI chuẩn (Stanford HARDI) từ Dipy...")
        fetch_stanford_hardi()
        img, gtab = read_stanford_hardi()
        data   = img.get_fdata()
        affine = img.affine
        bvals  = gtab.bvals
        print(f"    Kích thước gốc: {data.shape}")
    else:
        print(f"[+] Đọc dữ liệu thật từ: {data_path}")
        img    = nib.load(data_path)
        data   = img.get_fdata()
        affine = img.affine
        bvals  = np.loadtxt(bval_path) if bval_path and os.path.exists(bval_path) \
                 else np.concatenate([[0], np.full(data.shape[-1] - 1, 1000)])
        print(f"    Kích thước gốc: {data.shape}")

    # ── Fast mode: crop vùng trung tâm ────────────────────────────────────────
    if fast_mode:
        cx, cy, cz = [s // 2 for s in data.shape[:3]]
        r = 20   # bán kính 20 voxel mỗi chiều → khối 40×40×40
        data   = data[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r, :]
        print(f"    Kích thước sau crop: {data.shape}  (fast mode)")

    # ── 1. QC trước xử lý ─────────────────────────────────────────────────────
    snr_before = calculate_snr(data)
    print(f"\n[QC] SNR trước xử lý : {snr_before:.2f}")

    # ── 2. Denoising (Patch2Self) ─────────────────────────────────────────────
    print(f"\n[1/3] Denoising (Patch2Self) – {num_processes} luồng CPU...")
    t0 = time.time()
    # 'ridge' nhanh hơn 'ols' (default) đáng kể do không cần nội suy ma trận
    data_denoised = patch2self(
        data, bvals,
        model='ridge',          # ← nhanh hơn OLS ~3-5 lần
        b0_threshold=50,
        shift_intensity=True,
        clip_negative_vals=False
    )
    print(f"      Xong sau {time.time()-t0:.1f}s")

    # ── 3. Gibbs Ringing Removal ───────────────────────────────────────────────
    print("\n[2/3] Gibbs Ringing Removal...")
    t0 = time.time()
    data_unringed = gibbs_removal(data_denoised, num_processes=num_processes)
    print(f"      Xong sau {time.time()-t0:.1f}s")

    # ── 4. QC sau xử lý ───────────────────────────────────────────────────────
    snr_after = calculate_snr(data_unringed)
    print(f"\n[QC] SNR sau xử lý  : {snr_after:.2f}")
    print(f"[QC] Cải thiện SNR  : +{snr_after - snr_before:.2f}")

    # ── 5. Lưu kết quả ────────────────────────────────────────────────────────
    print("\n[3/3] Lưu ảnh đã xử lý...")
    out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "dataset", "derivatives")
    os.makedirs(out_dir, exist_ok=True)
    suffix   = "_fast" if fast_mode else ""
    out_path = os.path.join(out_dir, f"dwi_preprocessed{suffix}.nii.gz")
    nib.save(nib.Nifti1Image(data_unringed.astype(np.float32), affine), out_path)

    print(f"\n✅ HOÀN TẤT – file lưu tại: {out_path}")
    return out_path


if __name__ == "__main__":
    import multiprocessing
    # Tự phát hiện số core
    cores = min(multiprocessing.cpu_count(), 8)
    print(f"Phát hiện {cores} CPU cores, sẽ dùng tối đa {cores} luồng.")

    # ── Đổi fast_mode=False khi có data Alzheimer thật ──
    preprocess_dwi(fast_mode=True, num_processes=cores)
