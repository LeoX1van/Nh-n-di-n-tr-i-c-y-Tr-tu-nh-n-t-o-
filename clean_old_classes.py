"""
clean_old_classes.py
----------------------
Thư mục dataset/train và dataset/test đang bị LẪN giữa:
    - Các lớp CŨ từ Fruits-360 (Apple 10, Banana 1, Cherry 2, ... hàng trăm lớp)
    - Các lớp MỚI vừa cắt ra từ dataset Roboflow (banana, mango, orange, ...)

Script này sẽ tự động:
    - Giữ lại trong dataset/train, dataset/test CHỈ các lớp có tên nằm
      trong danh sách NEW_CLASSES bên dưới.
    - Di chuyển (không xóa) mọi lớp khác sang thư mục dataset_fruits360_backup/
      để lưu trữ, phòng khi cần dùng lại.

CÁCH DÙNG:
    python clean_old_classes.py
"""

import os
import shutil

DATASET_DIR = "dataset"
BACKUP_DIR = "dataset_fruits360_backup"

# Danh sách các lớp MỚI (tên thư mục thật đã thấy trong dataset/train của bạn) -
# đây là các lớp cắt ra từ Roboflow, cần GIỮ LẠI.
NEW_CLASSES = {
    "atis", "avocado", "balimbing", "banana", "buko", "caimito", "chico",
    "dragon fruit", "durian", "guava", "guyabano", "jackfruit", "lansones",
    "lemon", "mango", "mangosteen", "marang", "orange", "papaya",
    "pineapple", "rambutan", "rose apple", "sampalok", "santol",
    "siniguelas", "strawberry", "watermelon",
}


def clean_split(split_name):
    split_dir = os.path.join(DATASET_DIR, split_name)
    if not os.path.isdir(split_dir):
        print(f"   (bỏ qua: không có {split_dir})")
        return 0, 0

    backup_split_dir = os.path.join(BACKUP_DIR, split_name)
    os.makedirs(backup_split_dir, exist_ok=True)

    kept, moved = 0, 0
    for folder_name in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        if folder_name in NEW_CLASSES:
            kept += 1
            continue

        # Không thuộc danh sách mới -> chuyển sang backup
        dest_path = os.path.join(backup_split_dir, folder_name)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)  # tránh lỗi nếu backup đã tồn tại từ lần chạy trước
        shutil.move(folder_path, dest_path)
        moved += 1

    return kept, moved


def main():
    print(">> Bắt đầu dọn dẹp dataset...")
    for split in ("train", "test"):
        kept, moved = clean_split(split)
        print(f"   {split}: giữ lại {kept} lớp mới, chuyển {moved} lớp cũ sang backup")

    print(f"\n>> HOÀN TẤT! Các lớp cũ (Fruits-360) đã được chuyển sang "
          f"thư mục '{BACKUP_DIR}/' để lưu trữ.")
    print(">> dataset/train và dataset/test giờ chỉ còn các lớp trái cây mới.")
    print(">> Giờ có thể chạy: python train_model.py")


if __name__ == "__main__":
    main()
