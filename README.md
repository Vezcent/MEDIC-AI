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

Đưa các tham số QC (từ Task 2) vào mô hình để kiểm tra xem chất lượng hình ảnh có tác động hay làm nhiễu sự liên kết giữa đặc trưng đồ thị và điểm nhận thức hay không.