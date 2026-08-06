# LNL / LNL-MoEx Model Training on GTSRB

Dự án huấn luyện và đánh giá tối ưu hóa mô hình Vision Transformer (LNL / LNL-MoEx) trên bộ dữ liệu biển báo giao thông Đức (GTSRB).

---

## 🚀 Hướng dẫn nhanh dành cho người khác (Quick Start)

### 1. Clone Repository về máy
Do file trọng số mô hình `checkpoints/lnl_s_run_1_best_val.pth` (272MB) được lưu trữ qua **Git LFS**, bạn hãy clone và kéo file LFS bằng lệnh:

```bash
git clone https://github.com/hoangloc2011/LNL-GTSRB-Model.git
cd LNL-GTSRB-Model

# Kéo file checkpoint thực sự về (bắt buộc)
git lfs pull
```

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Chạy Mô hình

#### Option A: Chạy Test Nhanh (Dành cho kiểm thử trong 1 - 2 phút)
```bash
python train.py --quick
```

#### Option B: Chạy Đầy Đủ
```bash
python train.py
```

> **Ghi chú:**
> - Code sẽ **tự động tải bộ dữ liệu GTSRB**, giải nén và phân chia tập Train (90%) / Validation (10%) / Test (100%).
> - Tự động tạo thư mục `./checkpoints` lưu lại trọng số mô hình tốt nhất (Best Validation Model).
