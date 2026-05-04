
# Graph analysis of DTI images

Graph Analysis of Diffusion Tensor Imaging (DTI) for Brain Connectivity Study

## 🧠 Overview
- This project implements a pipeline for transforming Diffusion Tensor Imaging (DTI) data into brain connectivity graphs.

- Starting from raw diffusion MRI data, the pipeline performs preprocessing, tensor estimation, and tractography to reconstruct white matter pathways. The brain is then represented as a graph, where regions are nodes and structural connections between them are edges.

- From the resulting connectivity matrix, graph-based features such as degree, clustering coefficient, path length, and global efficiency are extracted. The main goal is to provide a structured workflow for generating brain graphs and network features from DTI data.

## 🎯 Objectives
- Build a pipeline from raw DTI data to brain connectivity graphs
- Perform preprocessing, tensor estimation, and tractography
- Construct adjacency matrices representing structural connectivity
- Extract graph-based features (degree, clustering, path length, etc.)
- Provide reusable outputs for further analysis or downstream tasks

## 🚀 How to Run

### 📌 System Requirements
- Python ≥ 3.10  
- Windows 10/11 (tested)  
- RAM ≥ 8GB  
- Free disk space ≥ 5GB  

---

### 0️⃣ Activate Virtual Environment

**Windows PowerShell**
```bash
.\.venv\Scripts\Activate.ps1
```
**Windows CMD**
```bash
.venv\Scripts\activate.bat
```

### 1️⃣ Install Requirements lib
- Command:
```bash
pip install -r requirements.txt
```
### 2️⃣ Download Sample Dataset
```bash
python scripts/download_dataset.py
```
- This will download the sample dataset (ds002737) into:
```bash
dataset/sub-01/
```
### 3️⃣ Run DTI Preprocessing 
- Command:
```bash
python scripts/task2_preprocess.py
```
* This step will:
    - Load Stanford HARDI DTI sample data
    - Apply preprocessing (denoising + Gibbs correction)
    - Compute SNR before/after processing
    - Save output to:
    ```bash
    dataset/derivatives/dwi_preprocessed_fast.nii.gz
    ```
### ⚙️ Optional Settings
#### Run full brain (no cropping):
- Open 
```bash
scripts/task2_preprocess.py
```
- Find script:
```bash
preprocess_dwi(fast_mode=True, num_processes=cores)
```
- Edit script:
```bash
preprocess_dwi(fast_mode=False, num_processes=cores)
```
#### Use real Alzheimer DTI data:
- Modify function call:
```bash
preprocess_dwi(
    data_path="path/to/sub-XX_dwi.nii.gz",
    bval_path="path/to/sub-XX.bval",
    fast_mode=False
)
```
### 📊 Task Checklist
| Task | Description | Status |
| :--- | :--- | :--- |
| **Task 1** | Dataset setup | ✅ **Done** |
| **Task 2** | DTI preprocessing | ✅ **Done** (sample mode) |
| **Task 3** | Tractography & connectome | ⏳ *TODO* |
| **Task 4** | Graph feature extraction | ⏳ *TODO* |
| **Task 5** | Statistical analysis | ⏳ *TODO* |
