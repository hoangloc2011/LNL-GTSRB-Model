import os
import sys
import shutil
import time
import glob
import re
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Đảm bảo Python tìm thấy thư mục dự án Locality-iN-Locality
# ---------------------------------------------------------
PROJECT_DIR = os.path.abspath('./Locality-iN-Locality')
if not os.path.exists(PROJECT_DIR):
    print("Đang tự động tải mã nguồn kiến trúc mô hình Locality-iN-Locality...")
    os.system("git clone https://github.com/Omid-Nejati/Locality-iN-Locality.git")

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks import PGD, FGSM
from ptflops import get_model_complexity_info

print("=== KIỂM TRA MÔI TRƯỜNG ===")
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Torchattacks version:", torchattacks.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị tính toán:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------
# 2. Tự động sửa lỗi cú pháp 'return' trong LNL.py
# ---------------------------------------------------------
lnl_file_path = os.path.join(PROJECT_DIR, "LNL.py")
if os.path.exists(lnl_file_path):
    with open(lnl_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    fixed = False
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip() == "return":
            lines[idx] = "    return model\n"
            fixed = True
            print(f"✓ Đã tự động sửa thành công dòng {idx+1} trong LNL.py thành 'return model'!")
            break
    if fixed:
        with open(lnl_file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

# Import mô hình sau khi sửa file
from LNL import LNL_S
from LNL_MoEx import LNL_MoEx_S

# ---------------------------------------------------------
# 3. Tải và giải nén dữ liệu GTSRB
# ---------------------------------------------------------
data_dir = './data'
gtsrb_dir = './data/GTSRB'
os.makedirs(data_dir, exist_ok=True)

def download_file(url, filename):
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        if zipfile.is_zipfile(filepath):
            print(f"File {filename} đã tồn tại và hợp lệ.")
            return
        else:
            print(f"File {filename} bị hỏng, đang xóa để tải lại...")
            try:
                os.remove(filepath)
            except Exception:
                pass

    print(f"Đang tải {filename}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Tải {filename} thành công.")

download_file("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip", "GTSRB_Final_Training_Images.zip")
download_file("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_Images.zip")
download_file("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip", "GTSRB_Final_Test_GT.zip")

def extract_zip(filename, dest):
    filepath = os.path.join(data_dir, filename)
    print(f"Đang giải nén {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest)
    print("Giải nén hoàn tất.")

if not os.path.exists(os.path.join(data_dir, "GTSRB/Final_Training")):
    extract_zip("GTSRB_Final_Training_Images.zip", data_dir)
if not os.path.exists(os.path.join(data_dir, "GTSRB/Final_Test")):
    extract_zip("GTSRB_Final_Test_Images.zip", data_dir)
if not os.path.exists(os.path.join(data_dir, "GT-final_test.csv")):
    extract_zip("GTSRB_Final_Test_GT.zip", data_dir)

# ---------------------------------------------------------
# 4. Tổ chức thư mục test và chia tập Train (90%) - Val (10%)
# ---------------------------------------------------------
test_dir = os.path.join(gtsrb_dir, 'test')
images_dir = os.path.join(gtsrb_dir, 'Final_Test/Images')
csv_path = os.path.join(data_dir, 'GT-final_test.csv')

if not os.path.exists(test_dir):
    os.makedirs(test_dir, exist_ok=True)
    print("Đang cấu hình tập Test...")
    with open(csv_path) as f:
        image_names = f.readlines()
    for text in image_names[1:]:
        parts = text.split(';')
        classes = int(parts[-1])
        image_name = parts[0]
        test_class_dir = os.path.join(test_dir, f"{classes:04d}")
        os.makedirs(test_class_dir, exist_ok=True)
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            shutil.copy(image_path, test_class_dir)

train_dir = os.path.join(gtsrb_dir, 'train')
val_dir = os.path.join(gtsrb_dir, 'val')
src_train_dir = os.path.join(gtsrb_dir, 'Final_Training/Images')

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("Đang phân chia tập Train (90%) và Val (10%)...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    base_dataset = dsets.ImageFolder(root=src_train_dir)
    val_size = int(len(base_dataset) * 0.10)
    train_size = len(base_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = torch.utils.data.random_split(base_dataset, [train_size, val_size], generator=generator)

    def copy_split_files(split, dest_folder):
        for idx in split.indices:
            img_path, label = split.dataset.samples[idx]
            class_name = f"{label:04d}"
            class_dest_dir = os.path.join(dest_folder, class_name)
            os.makedirs(class_dest_dir, exist_ok=True)
            shutil.copy(img_path, class_dest_dir)

    copy_split_files(val_split, val_dir)
    copy_split_files(train_split, train_dir)
    print("Đã hoàn tất phân chia tập dữ liệu!")

# ---------------------------------------------------------
# 5. Cấu hình Tham số & Khởi tạo DataLoaders
# ---------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Huấn luyện mô hình LNL-S trên GTSRB")
parser.add_argument("--quick", action="store_true", help="Kích hoạt chế độ test nhanh (1 Epoch, img_size 112) trong 1-2 phút")
args = parser.parse_args()

if args.quick:
    print("⚡ ĐÃ KÍCH HOẠT CHẾ ĐỘ TEST NHANH (--quick): Running 1 Epoch, img_size=112...")
    img_size = 112
    num_epochs = 1
    warmup_epochs = 1
    cooldown_epochs = 0
else:
    img_size = 224
    num_epochs = 50
    warmup_epochs = 5
    cooldown_epochs = 10

batch_size = 16   # Batch size cho GPU local 6GB VRAM
run_id = 1

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = dsets.ImageFolder(root=train_dir, transform=train_transform)
valset = dsets.ImageFolder(root=val_dir, transform=test_transform)
testset = dsets.ImageFolder(root=test_dir, transform=test_transform)

use_pin_memory = torch.cuda.is_available()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pin_memory, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=8, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=8, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

print(f"Kích thước dữ liệu: Train={len(trainset)} | Val={len(valset)} | Test={len(testset)}")

# ---------------------------------------------------------
# 6. Khởi tạo Mô hình LNL-S
# ---------------------------------------------------------
model = LNL_S(pretrained=False, img_size=img_size)
model.head = nn.Linear(in_features=model.head.in_features, out_features=43, bias=True)
model = model.to(device)

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=43)
loss_fn_mixup = SoftTargetCrossEntropy()
loss_fn_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_fn_ce_clean = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

# ---------------------------------------------------------
# 7. Quản lý Checkpoints & Quyết định Train/Test
# ---------------------------------------------------------
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
start_epoch = 0
best_val_acc = 0.0
has_completed_model = False

checkpoint_path = None

# 1. Kiểm tra file best_val.pth trước
best_val_path = os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_best_val.pth")
if os.path.exists(best_val_path):
    checkpoint_path = best_val_path

# 2. Nếu không có best_val, tìm file epoch_*.pth lớn nhất
if not checkpoint_path:
    checkpoint_pattern = os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        epochs = []
        for f in checkpoint_files:
            match = re.search(r'epoch_(\d+)\.pth$', f)
            if match:
                epochs.append((int(match.group(1)), f))
        if epochs:
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            checkpoint_path = latest_file

# 3. Kiểm tra file _latest.pth
if not checkpoint_path:
    latest_path = os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_latest.pth")
    if os.path.exists(latest_path):
        checkpoint_path = latest_path

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Đang khôi phục checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'best_val_acc' in checkpoint:
        best_val_acc = checkpoint['best_val_acc']
    
    # Nếu nạp từ file best_val.pth HOẶC epoch đã đạt đến num_epochs, chuyển thẳng sang Test
    if "best_val" in checkpoint_path or start_epoch >= num_epochs:
        has_completed_model = True
        print(f"✓ Đã nạp thành công Checkpoint tốt nhất (Val Acc: {best_val_acc:.2f}%)! Bỏ qua huấn luyện và chuyển thẳng sang Đánh giá tập Test...")
    else:
        print(f"✓ Khôi phục thành công từ Epoch {start_epoch}! Tiếp tục huấn luyện các epoch còn lại (Val Acc tốt nhất trước đó: {best_val_acc:.2f}%)")
else:
    print("Không tìm thấy checkpoint. Bắt đầu huấn luyện mới từ Epoch 0.")

# ---------------------------------------------------------
# 8. Vòng lặp Huấn luyện (Train Loop)
# ---------------------------------------------------------
if __name__ == '__main__':
    if not has_completed_model:
        print("\n================ BẮT ĐẦU HUẤN LUYỆN LNL-S ================")
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            start_time = time.time()
            use_mixup = (epoch < num_epochs - cooldown_epochs)

            if epoch == num_epochs - cooldown_epochs:
                print(f"\n--- Bắt đầu giai đoạn Cooldown: Huấn luyện ảnh sạch không dùng Label Smoothing trong {cooldown_epochs} epoch ---")

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast('cuda'):
                        if use_mixup:
                            images_mixed, labels_mixed = mixup_fn(images, labels)
                            outputs = model(images_mixed)
                            loss = loss_fn_mixup(outputs, labels_mixed)
                            _, predicted = torch.max(outputs.data, 1)
                            _, target_max = torch.max(labels_mixed, dim=-1)
                            correct += (predicted == target_max).sum().item()
                        else:
                            outputs = model(images)
                            loss = loss_fn_ce_clean(outputs, labels)
                            _, predicted = torch.max(outputs.data, 1)
                            correct += (predicted == labels).sum().item()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = loss_fn_ce_clean(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()

                total_loss += loss.item() * images.size(0)
                total += labels.size(0)

                if (i + 1) % 200 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] - Step [{i+1}/{len(train_loader)}] - Loss: {loss.item():.4f} - Acc: {(correct/total)*100:.2f}%")

            scheduler.step()
            epoch_loss = total_loss / total
            epoch_acc = (correct / total) * 100

            # Đánh giá tập Val
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()
            
            val_acc = (val_correct / val_total) * 100
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% - Val Acc: {val_acc:.2f}% - Time: {elapsed_time:.1f}s")

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                print(f"--> [BEST] Val Acc mới cao nhất: {best_val_acc:.2f}%! Đang lưu checkpoint best model...")

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }

            if is_best:
                torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_best_val.pth"))
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_latest.pth"))

            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                torch.save(checkpoint_state, os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_epoch_{epoch+1}.pth"))

    # ---------------------------------------------------------
    # 9. Đánh giá Test & FGSM Attack
    # ---------------------------------------------------------
    print("\n================ ĐÁNH GIÁ TRÊN TẬP TEST ================")
    best_val_path = os.path.join(checkpoint_dir, f"lnl_s_run_{run_id}_best_val.pth")
    if os.path.exists(best_val_path):
        checkpoint = torch.load(best_val_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f' Độ chính xác chuẩn (Standard Accuracy): {(100.0 * correct / total):.2f}%')
