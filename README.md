<<<<<<< HEAD
Task 1: Setup hệ thống & Lấy dữ liệu OpenNeuro

Tìm kiếm và tải bộ dữ liệu liên quan đến Alzheimer từ OpenNeuro (đảm bảo có chứa dữ liệu DWI/DTI). DWI/DTI là các kỹ thuật hình ảnh liên quan đến sự khuếch tán của phân tử nước trong não, hỗ trợ nghiên cứu cấu trúc chất trắng và các bó sợi thần kinh.  

Cài đặt các công cụ tiền xử lý (FSL, MRtrix3), chạy venv.

Task 2: Tiền xử lý ảnh & Định hướng Biomarker (RQ1)
Làm sạch ảnh MRI khuếch tán (Denoising, Eddy correction) và trích xuất các tham số chất lượng (QC).

Định hướng phân tích tập trung vào yếu tố Neurodegeneration (thoái hóa thần kinh), tương ứng với chữ N trong mô hình biomarker ATN. Quá trình thoái hóa này làm tổn thương tế bào não, dẫn đến teo não và có thể làm thay đổi cấu trúc cũng như sự kết nối giữa các vùng não.

Task 3: Xây dựng Đồ thị Não (Tractography & Parcellation)
Tiến hành phân vùng não, trong đó mỗi vùng não sẽ được xem là một node (nút) trong mạng lưới.  

Chạy thuật toán Tractography để thiết lập các edge (cạnh kết nối) giữa các node.  

Gán trọng số cho các edge dựa trên số lượng bó sợi thần kinh nối giữa hai vùng, hoặc sử dụng các chỉ số khuếch tán như FA, MD, AD, RD.

Task 4: Trích xuất Đặc trưng Đồ thị (Graph Features)

Tính toán các chỉ số mạng để mô tả cách não tổ chức và truyền thông tin.  

Viết code (Python/C++) để trích xuất 5 đặc trưng cốt lõi sau:

Degree: Đo lường xem một node có bao nhiêu kết nối với các node khác.  

Clustering coefficient: Phản ánh mức độ các vùng não lân cận có xu hướng kết nối thành cụm.  

Path length: Thể hiện độ dài đường đi trung bình giữa các node trong mạng.  

Global efficiency: Cho biết hiệu quả truyền thông tin toàn mạng.  

Small-worldness: Phản ánh đặc điểm mạng vừa có khả năng xử lý cục bộ tốt, vừa có khả năng truyền thông tin toàn cục hiệu quả.

Task 5: Phân tích Thống kê (RQ2)

Thu thập điểm số từ các bài kiểm tra nhận thức (cognition) như thang đo MMSE hoặc ADAS-Cog để làm biến kết quả. Các bài kiểm tra này phản ánh khả năng trí nhớ, tư duy, ngôn ngữ và xử lý thông tin của người bệnh

Chạy mô hình hồi quy để phân tích xem sự thay đổi trong cấu trúc hoặc kết nối não (thông qua các chỉ số graph ở Task 4) có liên quan như thế nào đến sự suy giảm nhận thức trong bệnh Alzheimer.

Đưa các tham số QC (từ Task 2) vào mô hình để kiểm tra xem chất lượng hình ảnh có tác động hay làm nhiễu sự liên kết giữa đặc trưng đồ thị và điểm nhận thức hay không
=======

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

#### Run full brain (no cropping)

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

#### Use real Alzheimer DTI data

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

>>>>>>> 595de4b17f0680de0e3dd01a117ce3f69704f66a
