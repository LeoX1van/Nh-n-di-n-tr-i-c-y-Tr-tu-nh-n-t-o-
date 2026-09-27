# Ứng dụng Nhận diện & Phân loại Trái cây

Gồm 2 phần:
1. `train_model.py` — huấn luyện mô hình CNN từ dataset ảnh trái cây.
2. `fruit_app.py` — ứng dụng giao diện (Tkinter, giao diện chỉnh chu dạng sidebar), hỗ trợ **chọn ảnh**, **camera laptop**, và **camera điện thoại kết nối qua cáp**.

---

## Bước 1: Mở dự án trong VSCode

1. Tạo 1 thư mục mới, ví dụ `fruit-recognition`, copy 4 file (`train_model.py`, `fruit_app.py`, `requirements.txt`, `README.md`) vào đó.
2. Mở VSCode → **File > Open Folder...** → chọn thư mục `fruit-recognition`.
3. Cài extension **Python** (Microsoft) trong VSCode nếu chưa có (mục Extensions, tìm "Python").
4. Mở Terminal trong VSCode: menu **Terminal > New Terminal** (hoặc `Ctrl + ` `).

---

## Bước 2: Cài Python và thư viện

Yêu cầu Python 3.9–3.11 (khuyến nghị 3.10).

Trong Terminal của VSCode (đang đứng ở thư mục dự án), chạy:

```bash
pip install -r requirements.txt
```

> Nếu máy không có GPU, TensorFlow vẫn chạy được trên CPU, chỉ train chậm hơn.

---

## Bước 3: Chuẩn bị dataset

Tải dataset trái cây, ví dụ **Fruits-360** trên Kaggle:
https://www.kaggle.com/datasets/moltean/fruits

Giải nén và sắp xếp theo cấu trúc (đặt trong thư mục dự án):

```
fruit-recognition/
  dataset/
    train/
      Apple/
        img001.jpg
        ...
      Banana/
        ...
    test/
      Apple/
        ...
      Banana/
        ...
  train_model.py
  fruit_app.py
  ...
```

Tên thư mục con = tên nhãn (loại trái cây) mô hình sẽ học.

---

## Bước 4: Huấn luyện mô hình

> **Bản cập nhật:** `train_model.py` giờ dùng kỹ thuật **Transfer Learning với MobileNetV2** thay vì train CNN từ đầu — model sẽ nhận diện chính xác hơn nhiều với ảnh thực tế và qua camera, vì tận dụng được kiến thức MobileNetV2 đã học từ hàng triệu ảnh thật (ImageNet). Cách chạy vẫn y hệt như cũ.

Trong Terminal VSCode:

```bash
python train_model.py
```

Quá trình train có 2 giai đoạn (script tự động chạy lần lượt, không cần làm gì thêm):
- **Giai đoạn 1:** chỉ train phần "đầu" phân loại mới thêm vào (nhanh hơn).
- **Giai đoạn 2 (fine-tune):** mở khóa một phần MobileNetV2 để tinh chỉnh sâu hơn (chậm hơn 1 chút nhưng chính xác hơn).

Sau khi chạy xong sẽ có 2 file mới trong thư mục dự án:
- `fruit_model.h5`
- `class_indices.json`

> Lưu ý: nếu trước đó bạn đã train bằng bản cũ (không dùng MobileNetV2), phải **train lại từ đầu bằng file train_model.py mới này**, và dùng đúng bản `fruit_app.py` mới (đã cập nhật để khớp kích thước ảnh 224x224) — 2 bản cũ/mới không tương thích với nhau.

---

## Bước 5: Chạy ứng dụng

```bash
python fruit_app.py
```

Hoặc mở file `fruit_app.py` trong VSCode rồi bấm nút **▷ Run** ở góc trên bên phải.

Giao diện gồm:
- Sidebar bên trái: nút chọn ảnh, chọn nguồn camera, nút bật/tắt camera.
- Khung lớn bên phải: hiển thị ảnh/video.
- Thẻ kết quả bên dưới: tên trái cây + % độ tin cậy.

---

## Kết nối camera điện thoại qua cáp (thay cho webcam laptop)

Cách đơn giản nhất là dùng app **IP Webcam** (Android, miễn phí) kết hợp **USB tethering** — điện thoại và máy tính sẽ "nói chuyện" qua cáp USB thay vì cần WiFi.

**Trên điện thoại:**
1. Cài app **IP Webcam** từ Google Play (tác giả: Pavel Khlebovich).
2. Cắm cáp USB nối điện thoại với máy tính.
3. Vào **Cài đặt > Mạng & Internet > Điểm phát sóng & Chia sẻ kết nối > Chia sẻ kết nối qua USB** (tên mục có thể khác chút tùy hãng điện thoại) → bật lên. Bước này giúp máy tính nhận điện thoại như 1 kết nối mạng qua cáp.
4. Mở app **IP Webcam** → kéo xuống cuối → bấm **Start server**.
5. App sẽ hiển thị dòng chữ dạng: `http://192.168.42.129:8080` — đây chính là địa chỉ camera của bạn.

**Trên máy tính (trong app `fruit_app.py`):**
1. Ở sidebar, chọn **"Camera điện thoại (qua cáp)"**.
2. Nhập vào ô địa chỉ: `http://<IP_hiện_trên_điện_thoại>:8080/video` (nhớ thêm `/video` ở cuối).
   Ví dụ: `http://192.168.42.129:8080/video`
3. Bấm **Bật Camera**.

> Nếu dùng iPhone: có thể dùng app tương tự như "IP Camera Lite" (App Store), cách làm tương tự — mở app, lấy URL dạng `http://<ip>:port/video` rồi nhập vào ô địa chỉ.

> Nếu không bật được USB tethering, bạn vẫn có thể dùng cách cũ: kết nối điện thoại và máy tính vào **chung 1 mạng WiFi**, rồi lấy URL y hệt như trên — không cần cáp, chỉ khác là không "chỉnh chu" như yêu cầu qua cáp nhưng vẫn hoạt động tốt.

---

## Một số lưu ý khi làm báo cáo / bảo vệ đồ án

- Kiến trúc CNN: 3 lớp Conv2D + MaxPooling → Flatten → Dense → Dropout → Dense (softmax).
- Nên báo cáo thêm: số lượng ảnh train/test, độ chính xác (accuracy) đạt được.
- Có thể nâng cấp bằng transfer learning (MobileNetV2 pretrained) nếu muốn tăng độ chính xác.
- Camera điện thoại thường cho ảnh nét hơn, ổn định ánh sáng tốt hơn webcam laptop — phù hợp để demo trực tiếp.

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân / Cách khắc phục |
|---|---|
| `Không tìm thấy thư mục dataset/train` | Kiểm tra lại đường dẫn và cấu trúc thư mục dataset |
| Không mở được camera laptop | Kiểm tra camera có bị ứng dụng khác (Zoom, Teams...) chiếm dụng không |
| Không mở được camera điện thoại | Kiểm tra: đã bật USB tethering / cùng mạng chưa, app IP Webcam đã "Start server" chưa, URL có đúng và có `/video` ở cuối không |
| Cài `tensorflow` báo lỗi | Kiểm tra phiên bản Python (nên dùng 3.10), thử `pip install tensorflow-cpu` nếu không có GPU |
| Dự đoán sai nhiều | Dataset ít ảnh / không đa dạng góc chụp, ánh sáng — nên bổ sung ảnh và train lại |
