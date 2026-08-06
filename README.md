# LNL / LNL-MoEx Model Training on GTSRB
## 🚀 Hướng dẫn nhanh

### 1. Clone Repository về máy

### 2. Cài đặt Môi trường
Tạo và kích hoạt môi trường ảo Python (Python 3.10 - 3.12):

#### Trên Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

#### Cài đặt thư viện phụ thuộc:
pip install -r requirements.txt

#### Cài đặt PyTorch hỗ trợ GPU CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


### 3. Chạy Huấn luyện Mô hình
 Chạy Huấn luyện Đầy Đủ (50 Epochs - Đạt độ chính xác >99.7%)
```bash
python train.py
```

> **Ghi chú:**
> - Code sẽ **tự động tải bộ dữ liệu GTSRB**, giải nén và phân chia tập Train (90%) / Validation (10%) / Test (100%).
> - Tự động tạo thư mục `./checkpoints` lưu lại trọng số mô hình tốt nhất (Best Validation Model).
