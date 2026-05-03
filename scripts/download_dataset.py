import os
import sys
import subprocess

def install_openneuro():
    try:
        import openneuro
    except ImportError:
        print("Đang cài đặt thư viện openneuro-py...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        print("Cài đặt thành công!")

def download_dataset():
    install_openneuro()
    import openneuro
    
    # Thư mục đích
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    # Ở đây chúng ta dùng tạm dataset ds002737 làm mẫu để có file nifti test.
    # (Lưu ý: Dataset OASIS-3 hay ADNI thường yêu cầu đăng ký tài khoản và tải qua web LONI/OASIS, không mở hoàn toàn trên OpenNeuro).
    dataset_id = "ds002737"
    
    print(f"Bắt đầu tải dataset mẫu {dataset_id} về {target_dir}...")
    print("Đang tải dữ liệu SAMPLE (sub-01)...")
    
    # Chỉ tải thư mục của 1 subject mẫu để tiết kiệm bộ nhớ (< 20GB)
    sample_subjects = ["sub-01"]
    openneuro.download(dataset=dataset_id, target_dir=target_dir, include=sample_subjects)
    print("Hoàn tất tải dữ liệu mẫu!")

if __name__ == "__main__":
    download_dataset()
