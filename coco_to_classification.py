"""
coco_to_classification.py
----------------------------
Chuyển 1 dataset định dạng COCO (object detection, tải từ Roboflow) sang
định dạng Classification (ảnh + tên lớp) mà train_model.py cần.

Cách hoạt động: đọc file _annotations.coco.json trong mỗi thư mục
train/valid/test, với mỗi vật thể (bounding box) được đánh dấu trong
ảnh, CẮT đúng vùng đó ra thành 1 ảnh riêng, rồi lưu vào:
    dataset/train/<tên_lớp>/...
    dataset/test/<tên_lớp>/...

CÁCH DÙNG:
1. Giải nén file .zip tải từ Roboflow (định dạng COCO JSON), đặt thư mục
   đó (ví dụ "Tropical fruits.v1i.coco") vào cùng thư mục dự án.
2. Sửa biến SOURCE_DIR bên dưới trỏ đúng tên thư mục vừa giải nén.
3. Chạy: python coco_to_classification.py
4. Kết quả sẽ tạo/thêm vào dataset/train và dataset/test.

LƯU Ý: Nếu thư mục dataset/train, dataset/test đã có sẵn dữ liệu từ
trước (VD Fruits-262), script này sẽ THÊM (bổ sung) các lớp trùng tên,
không xóa dữ liệu cũ - giúp tăng thêm ảnh thực tế cho các lớp đã có.
"""

import json
import os

from PIL import Image

# ====== CẤU HÌNH (sửa lại cho đúng máy bạn) ======
SOURCE_DIR = "Tropical fruits.v1i.coco"  # đổi thành đúng tên thư mục vừa giải nén
DEST_DIR = "dataset"                      # thư mục đích (giữ nguyên)

# Roboflow xuất ra 3 thư mục con: train, valid, test.
# Ánh xạ: train -> dataset/train, valid -> dataset/train (thêm data train),
#         test -> dataset/test
SPLIT_MAPPING = {
    "train": "train",
    "valid": "train",
    "test": "test",
}

MIN_BOX_SIZE = 20  # bỏ qua các khung quá nhỏ (px) - thường là nhiễu/lỗi gán nhãn
# ===================================================


def process_split(split_name, dest_split):
    split_dir = os.path.join(SOURCE_DIR, split_name)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")

    if not os.path.isfile(ann_path):
        print(f"   (bỏ qua '{split_name}': không tìm thấy _annotations.coco.json)")
        return 0

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # id -> tên lớp
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    # id ảnh -> tên file ảnh
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    count = 0
    for ann in coco["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        if not cat_name or cat_name.lower() in ("fruits", "objects"):
            continue  # bỏ qua lớp "cha" tổng quát nếu Roboflow có tạo thêm

        img_file = img_id_to_file.get(ann["image_id"])
        if not img_file:
            continue

        img_path = os.path.join(split_dir, img_file)
        if not os.path.isfile(img_path):
            continue

        x, y, w, h = ann["bbox"]
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                cropped = im.crop((int(x), int(y), int(x + w), int(y + h)))

                dest_folder = os.path.join(DEST_DIR, dest_split, cat_name)
                os.makedirs(dest_folder, exist_ok=True)

                out_name = f"{os.path.splitext(img_file)[0]}_{ann['id']}.jpg"
                cropped.save(os.path.join(dest_folder, out_name), quality=90)
                count += 1
        except Exception as e:
            print(f"   Lỗi xử lý ảnh {img_file}: {e}")

    return count


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise SystemExit(
            f"Không tìm thấy thư mục '{SOURCE_DIR}'.\n"
            "Hãy sửa biến SOURCE_DIR trong file này trỏ đúng tới thư mục "
            "vừa giải nén từ Roboflow."
        )

    print(">> Bắt đầu chuyển đổi COCO -> Classification...")
    totals = {"train": 0, "test": 0}

    for split_name, dest_split in SPLIT_MAPPING.items():
        n = process_split(split_name, dest_split)
        totals[dest_split] = totals.get(dest_split, 0) + n
        print(f"   {split_name} -> dataset/{dest_split}: {n} ảnh đã cắt")

    print(f"\n>> HOÀN TẤT! Tổng: {totals.get('train', 0)} ảnh train, "
          f"{totals.get('test', 0)} ảnh test.")
    print(">> Giờ có thể chạy: python train_model.py")


if __name__ == "__main__":
    main()
