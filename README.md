# LNL / LNL-MoEx Model Training on GTSRB

Dự án huấn luyện và đánh giá tối ưu hóa mô hình Vision Transformer (LNL / LNL-MoEx) trên bộ dữ liệu biển báo giao thông Đức (GTSRB).

---

## 🚀 Hướng dẫn nhanh dành cho người khác (Quick Start)

### 1. Clone Repository về máy
```bash
git clone --recursive <URL_REPO_CỦA_BẠN>
cd <TÊN_THƯ_MỤC_REPO>
```
*(Nếu đã clone mà quên `--recursive`, chạy thêm: `git submodule update --init --recursive` hoặc `git clone https://github.com/Omid-Nejati/Locality-iN-Locality.git`)*

### 2. Cài đặt Môi trường
Tạo và kích hoạt môi trường ảo Python (Python 3.10 - 3.12):

#### Trên Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Cài đặt thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

#### Cài đặt PyTorch hỗ trợ GPU CUDA (Khuyên dùng cho máy có GPU NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Chạy Huấn luyện Mô hình

#### Options A: Chạy Test Nhanh Siêu Tốc (Dành cho kiểm thử code trong 1 - 2 phút)
```bash
python train.py --quick
```

#### Options B: Chạy Huấn luyện Đầy Đủ (50 Epochs - Đạt độ chính xác >99.7%)
```bash
python train.py
```

> **Ghi chú:**
> - Code sẽ **tự động tải bộ dữ liệu GTSRB**, giải nén và phân chia tập Train (90%) / Validation (10%) / Test (100%).
> - Tự động tạo thư mục `./checkpoints` lưu lại trọng số mô hình tốt nhất (Best Validation Model).
