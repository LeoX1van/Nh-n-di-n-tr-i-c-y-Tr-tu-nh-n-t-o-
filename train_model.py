"""
train_model.py (phiên bản Transfer Learning - MobileNetV2)
-------------------------------------------------------------
Huấn luyện mô hình nhận diện trái cây bằng cách tận dụng MobileNetV2
(đã học sẵn trên hàng triệu ảnh thực tế - ImageNet), giúp nhận diện
tốt hơn nhiều với ảnh chụp thực tế / qua camera so với model tự xây
từ đầu.

CÁCH DÙNG: giống bản cũ, chỉ cần chạy:
    python train_model.py

Yêu cầu cấu trúc dataset/train/<loại>/... và dataset/test/<loại>/...
(xem README.md để biết chi tiết)

QUAN TRỌNG: file này thay thế train_model.py cũ. Sau khi train xong
bằng file này, phải dùng fruit_app.py phiên bản mới (đã cập nhật để
khớp với IMG_SIZE=224 và cách tiền xử lý ảnh của MobileNetV2).
"""

import json
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

# ====== CẤU HÌNH (có thể chỉnh sửa) ======
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)      # kích thước chuẩn của MobileNetV2
BATCH_SIZE = 32

EPOCHS_HEAD = 6            # giai đoạn 1: chỉ train phần đầu mới thêm
EPOCHS_FINETUNE = 6        # giai đoạn 2: fine-tune thêm 1 phần MobileNetV2
FINETUNE_AT_LAYER = 100    # mở khóa (train) từ layer thứ bao nhiêu trở đi

MODEL_PATH = "fruit_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
# ==========================================


def build_data_generators():
    """Data augmentation MẠNH hơn bản cũ để model quen nhiều điều kiện
    ánh sáng, góc chụp, nền ảnh khác nhau - giúp nhận diện tốt hơn qua
    camera và ảnh thực tế (không chỉ ảnh nền trắng studio)."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # chuẩn hóa đúng kiểu MobileNetV2
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        shear_range=0.1,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        vertical_flip=True,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # QUAN TRỌNG: lấy danh sách lớp CỐ ĐỊNH từ thư mục train, rồi ép cả
    # train và test dùng chung đúng danh sách này. Nếu không, khi thư mục
    # test thiếu 1 vài lớp (ít ảnh không đủ chia), Keras sẽ tạo ra 2 bộ
    # nhãn có số cột KHÁC NHAU giữa train/test -> lỗi khi train.
    class_names = sorted(
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=class_names,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )
    return train_gen, test_gen


def build_model(num_classes: int):
    """Xây dựng model = MobileNetV2 (đóng băng) + phần đầu phân loại mới."""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,       # bỏ phần phân loại gốc của ImageNet (1000 lớp)
        weights="imagenet",      # dùng trọng số đã học sẵn
    )
    base_model.trainable = False  # giai đoạn 1: đóng băng, không train lại phần này

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def main():
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        raise SystemExit(
            f"Không tìm thấy thư mục '{TRAIN_DIR}' hoặc '{TEST_DIR}'.\n"
            "Hãy tạo cấu trúc dataset/train/<loại_trái_cây>/... và "
            "dataset/test/<loại_trái_cây>/... trước khi chạy script này."
        )

    print(">> Đang nạp dữ liệu...")
    train_gen, test_gen = build_data_generators()
    num_classes = len(train_gen.class_indices)
    print(f">> Số lớp (loại trái cây): {num_classes}")

    print(">> Đang xây dựng model (MobileNetV2 + phần đầu phân loại)...")
    model, base_model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    # ===== GIAI ĐOẠN 1: train phần đầu mới thêm, giữ nguyên MobileNetV2 =====
    print("\n>> GIAI ĐOẠN 1: Train phần đầu phân loại (MobileNetV2 bị đóng băng)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # ===== GIAI ĐOẠN 2: fine-tune - mở khóa 1 phần MobileNetV2 để tinh chỉnh =====
    print("\n>> GIAI ĐOẠN 2: Fine-tune một phần MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:FINETUNE_AT_LAYER]:
        layer.trainable = False  # vẫn giữ đóng băng các layer đầu (đặc trưng cơ bản)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # learning rate rất nhỏ
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks,
    )

    print("\n>> Đánh giá trên tập test...")
    loss, acc = model.evaluate(test_gen)
    print(f">> Độ chính xác trên tập test: {acc * 100:.2f}%")

    print(f">> Lưu mô hình vào '{MODEL_PATH}'...")
    model.save(MODEL_PATH)

    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f">> Đã lưu bảng nhãn vào '{CLASS_INDEX_PATH}'.")

    print(">> HOÀN TẤT! Giờ bạn có thể chạy fruit_app.py để nhận diện trái cây.")


if __name__ == "__main__":
    main()
