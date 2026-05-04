import os
import sys
import subprocess

def install_openneuro():
    try:
        import openneuro
    except ImportError:
        print("Installing openneuro-py library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        print("Installation complete!")

def download_dataset():
    install_openneuro()
    import openneuro
    
    # Target directory
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    os.makedirs(target_dir, exist_ok=True)
    
    # Using dataset ds002737 as a sample to test the download pipeline.
    # (Note: OASIS-3 or ADNI datasets require account registration
    #  and must be downloaded via LONI/OASIS portal, not fully open on OpenNeuro.)
    dataset_id = "ds002737"
    
    print(f"Downloading sample dataset {dataset_id} to {target_dir}...")
    print("Downloading SAMPLE data (sub-01)...")
    
    # Download only 1 sample subject to save disk space (< 20GB)
    sample_subjects = ["sub-01"]
    openneuro.download(dataset=dataset_id, target_dir=target_dir, include=sample_subjects)
    print("Sample data download complete!")

if __name__ == "__main__":
    download_dataset()
