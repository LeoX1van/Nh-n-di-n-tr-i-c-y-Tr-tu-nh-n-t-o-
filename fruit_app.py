"""
fruit_app.py
-------------
Ứng dụng nhận diện & phân loại trái cây - giao diện chỉnh chu (ttk).

2 CHẾ ĐỘ NHẬN DIỆN:
  1. Chọn ảnh từ máy (Import ảnh)
  2. Camera trực tiếp - có thể chọn:
       a) Camera laptop (webcam mặc định)
       b) Camera điện thoại (qua app IP Webcam, kết nối bằng cáp USB tethering)
          -> xem hướng dẫn kết nối điện thoại trong README.md

YÊU CẦU TRƯỚC KHI CHẠY:
  - Đã chạy train_model.py và có 2 file: fruit_model.h5, class_indices.json
    (đặt cùng thư mục với file này)
  - Cài thư viện: pip install -r requirements.txt

CÁCH CHẠY (trong VSCode):
  - Mở thư mục dự án trong VSCode
  - Mở terminal (Ctrl + `) rồi chạy: python fruit_app.py
  - Hoặc bấm nút Run ▷ ở góc trên bên phải khi đang mở file này
"""

import json
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "fruit_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (224, 224)  # PHẢI giống với IMG_SIZE lúc train (bản MobileNetV2)

STABLE_SECONDS = 2.0        # phải giữ ổn định trong ngần này giây mới chốt kết quả
MIN_CONFIDENCE = 55.0       # dưới ngưỡng này coi như "chưa chắc", không tính vào ổn định
AIM_BOX_RATIO = 0.55        # kích thước khung ngắm so với cạnh nhỏ hơn của khung hình
DISPLAY_SIZE = (640, 480)   # kích thước CỐ ĐỊNH của khung xem ảnh/camera (không co giãn)

# ---------- Bảng màu giao diện ----------
COLOR_BG = "#f4f6f5"
COLOR_SIDEBAR = "#1f2937"
COLOR_ACCENT = "#2e7d32"
COLOR_ACCENT_DARK = "#1b5e20"
COLOR_CARD = "#ffffff"
COLOR_MUTED = "#6b7280"


class FruitRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nhận diện & Phân loại Trái cây")
        self.root.geometry("980x740")
        self.root.minsize(980, 700)
        self.root.configure(bg=COLOR_BG)

        self._load_model_and_labels()

        self.cap = None
        self.camera_running = False

        # Trạng thái "ngắm & giữ ổn định" khi nhận diện qua camera
        self.pending_label = None      # tên đang được nghi ngờ, chờ xác nhận ổn định
        self.pending_start_time = None  # thời điểm bắt đầu thấy nhãn này liên tục
        self.is_locked = False          # đã chốt kết quả cuối cùng chưa
        self.locked_label = None
        self.locked_confidence = None

        self._setup_style()
        self._build_ui()

    # ---------- Nạp mô hình ----------
    def _load_model_and_labels(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDEX_PATH):
            messagebox.showerror(
                "Thiếu file mô hình",
                f"Không tìm thấy '{MODEL_PATH}' hoặc '{CLASS_INDEX_PATH}'.\n"
                "Hãy chạy train_model.py trước để tạo mô hình.",
            )
            raise SystemExit(1)

        self.model = load_model(MODEL_PATH)
        with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.idx_to_class = {int(k): v for k, v in raw.items()}

    # ---------- Style chung ----------
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR)
        style.configure("Main.TFrame", background=COLOR_BG)

        style.configure(
            "SidebarTitle.TLabel", background=COLOR_SIDEBAR, foreground="white",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "SidebarText.TLabel", background=COLOR_SIDEBAR, foreground="#cbd5e1",
            font=("Segoe UI", 10),
        )
        style.configure(
            "ResultTitle.TLabel", background=COLOR_CARD, foreground=COLOR_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "ResultValue.TLabel", background=COLOR_CARD, foreground=COLOR_ACCENT_DARK,
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10,
            background=COLOR_ACCENT, foreground="white",
        )
        style.map("Accent.TButton", background=[("active", COLOR_ACCENT_DARK)])

        style.configure(
            "Secondary.TButton", font=("Segoe UI", 10, "bold"), padding=8,
            background="#374151", foreground="white",
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#4b5563"), ("disabled", "#1f2937")],
            foreground=[("disabled", "#6b7280")],
        )

        style.configure("TRadiobutton", background=COLOR_SIDEBAR, foreground="white",
                         font=("Segoe UI", 10))
        style.map("TRadiobutton", background=[("active", COLOR_SIDEBAR)])

    # ---------- Giao diện ----------
    def _build_ui(self):
        # ===== Sidebar (bên trái): điều khiển =====
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="🍎 Fruit AI", style="SidebarTitle.TLabel").pack(
            anchor="w", padx=20, pady=(24, 4)
        )
        ttk.Label(
            sidebar, text="Nhận diện & phân loại trái cây",
            style="SidebarText.TLabel"
        ).pack(anchor="w", padx=20, pady=(0, 20))

        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=8)

        # -- Nút chọn ảnh --
        ttk.Label(sidebar, text="NHẬN DIỆN TỪ ẢNH", style="SidebarText.TLabel").pack(
            anchor="w", padx=20, pady=(16, 6)
        )
        self.btn_import = ttk.Button(
            sidebar, text="📁  Chọn ảnh...", style="Accent.TButton",
            command=self.on_import_image
        )
        self.btn_import.pack(fill="x", padx=20, pady=4)

        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=16)

        # -- Nguồn camera --
        ttk.Label(sidebar, text="NHẬN DIỆN QUA CAMERA", style="SidebarText.TLabel").pack(
            anchor="w", padx=20, pady=(0, 6)
        )

        self.camera_source_var = tk.StringVar(value="laptop")
        ttk.Radiobutton(
            sidebar, text="Camera laptop", value="laptop",
            variable=self.camera_source_var, style="TRadiobutton"
        ).pack(anchor="w", padx=20, pady=2)
        ttk.Radiobutton(
            sidebar, text="Camera điện thoại (qua cáp)", value="phone",
            variable=self.camera_source_var, style="TRadiobutton"
        ).pack(anchor="w", padx=20, pady=2)

        ttk.Label(sidebar, text="Địa chỉ URL điện thoại:", style="SidebarText.TLabel").pack(
            anchor="w", padx=20, pady=(10, 2)
        )
        self.phone_url_var = tk.StringVar(value="http://192.168.42.129:8080/video")
        entry = tk.Entry(
            sidebar, textvariable=self.phone_url_var, font=("Segoe UI", 9),
            bg="#374151", fg="white", insertbackground="white", relief="flat"
        )
        entry.pack(fill="x", padx=20, pady=(0, 4), ipady=4)
        ttk.Label(
            sidebar, text="(Xem hướng dẫn lấy URL này trong README.md)",
            style="SidebarText.TLabel"
        ).pack(anchor="w", padx=20, pady=(0, 10))

        self.btn_camera = ttk.Button(
            sidebar, text="🎥  Bật Camera", style="Accent.TButton",
            command=self.on_toggle_camera
        )
        self.btn_camera.pack(fill="x", padx=20, pady=4)

        self.btn_rescan = ttk.Button(
            sidebar, text="🔄  Quét lại", style="Secondary.TButton",
            command=self.on_rescan, state="disabled"
        )
        self.btn_rescan.pack(fill="x", padx=20, pady=4)

        # -- Trạng thái mô hình --
        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=16)
        num_classes = len(self.idx_to_class)
        ttk.Label(
            sidebar, text=f"✔ Mô hình đã nạp\n{num_classes} loại trái cây",
            style="SidebarText.TLabel"
        ).pack(anchor="w", padx=20, pady=4)

        # ===== Khu vực chính (bên phải) =====
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(side="left", fill="both", expand=True)

        # Khung hiển thị ảnh/camera (dạng card) - KÍCH THƯỚC CỐ ĐỊNH, không co giãn
        # để khung kết quả bên dưới luôn hiển thị đầy đủ, không bị đẩy khuất.
        image_card = tk.Frame(
            main, bg=COLOR_CARD, highlightbackground="#e5e7eb", highlightthickness=1,
            width=DISPLAY_SIZE[0] + 4, height=DISPLAY_SIZE[1] + 4,
        )
        image_card.pack(padx=24, pady=(24, 12))
        image_card.pack_propagate(False)  # không cho nội dung bên trong làm đổi kích thước

        self.display_label = tk.Label(image_card, bg="#111827")
        self.display_label.pack(padx=2, pady=2, fill="both", expand=True)
        self._set_placeholder_image()

        # Khung kết quả (dạng card)
        result_card = tk.Frame(main, bg=COLOR_CARD, highlightbackground="#e5e7eb",
                                highlightthickness=1)
        result_card.pack(padx=24, pady=(0, 24), fill="x")

        inner = tk.Frame(result_card, bg=COLOR_CARD)
        inner.pack(padx=20, pady=16, fill="x")

        ttk.Label(inner, text="KẾT QUẢ NHẬN DIỆN", style="ResultTitle.TLabel").pack(anchor="w")
        self.result_var = tk.StringVar(value="Chưa có kết quả")
        ttk.Label(inner, textvariable=self.result_var, style="ResultValue.TLabel").pack(
            anchor="w", pady=(2, 0)
        )

    def _set_placeholder_image(self):
        placeholder = Image.new("RGB", DISPLAY_SIZE, color="#111827")
        imgtk = ImageTk.PhotoImage(placeholder)
        self.display_label.imgtk = imgtk
        self.display_label.configure(image=imgtk)

    # ---------- Đọc ảnh an toàn (hỗ trợ đường dẫn có dấu/ký tự đặc biệt) ----------
    def _read_image_safe(self, file_path):
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    # ---------- Xử lý ảnh chung ----------
    def _preprocess(self, frame_bgr):
        img = cv2.resize(frame_bgr, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        img = preprocess_input(img)  # chuẩn hóa đúng kiểu MobileNetV2 (không phải /255)
        img = np.expand_dims(img, axis=0)
        return img

    def _predict(self, frame_bgr):
        x = self._preprocess(frame_bgr)
        preds = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = self.idx_to_class.get(idx, "Không rõ")
        confidence = float(preds[idx]) * 100
        return label, confidence

    def _fit_frame(self, frame, target_w=DISPLAY_SIZE[0], target_h=DISPLAY_SIZE[1]):
        """Cắt (center-crop) và resize ảnh về đúng kích thước cố định target_w x target_h,
        giữ đúng tỉ lệ (không bị méo hình), để khung ngắm luôn khớp chính xác vị trí
        hiển thị trên màn hình."""
        h, w = frame.shape[:2]
        target_aspect = target_w / target_h
        src_aspect = w / h
        if src_aspect > target_aspect:
            new_w = int(h * target_aspect)
            x0 = (w - new_w) // 2
            cropped = frame[:, x0:x0 + new_w]
        else:
            new_h = int(w / target_aspect)
            y0 = (h - new_h) // 2
            cropped = frame[y0:y0 + new_h, :]
        return cv2.resize(cropped, (target_w, target_h))

    def _show_frame_on_label(self, frame_bgr):
        """Hiển thị 1 khung hình đã đúng kích thước DISPLAY_SIZE lên giao diện."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.display_label.imgtk = imgtk
        self.display_label.configure(image=imgtk)

    # ---------- Chế độ 1: Import ảnh ----------
    def on_import_image(self):
        if self.camera_running:
            self.on_toggle_camera()

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh trái cây",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        frame = self._read_image_safe(file_path)
        if frame is None:
            messagebox.showerror(
                "Lỗi",
                "Không đọc được ảnh này.\n"
                "File có thể bị hỏng hoặc không đúng định dạng ảnh (.jpg/.png/.bmp)."
            )
            return

        label, confidence = self._predict(frame)
        self.result_var.set(f"{label}   •   {confidence:.1f}% tin cậy")

        display_frame = self._fit_frame(frame)
        cv2.putText(
            display_frame, f"{label} ({confidence:.1f}%)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        self._show_frame_on_label(display_frame)

    # ---------- Chế độ 2: Camera trực tiếp ----------
    def _get_camera_source(self):
        """Trả về nguồn camera: 0 (laptop) hoặc URL (điện thoại)."""
        if self.camera_source_var.get() == "phone":
            url = self.phone_url_var.get().strip()
            if not url:
                messagebox.showerror(
                    "Thiếu địa chỉ", "Vui lòng nhập URL camera điện thoại."
                )
                return None
            return url
        return 0

    def on_toggle_camera(self):
        if not self.camera_running:
            source = self._get_camera_source()
            if source is None:
                return

            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Lỗi kết nối camera",
                    "Không mở được camera. Kiểm tra:\n"
                    "- Nếu dùng camera laptop: camera có đang bị app khác chiếm dụng không?\n"
                    "- Nếu dùng camera điện thoại: URL đã đúng chưa, điện thoại đã bật "
                    "app IP Webcam và kết nối cáp/USB tethering chưa? (xem README.md)"
                )
                self.cap = None
                return

            self.camera_running = True
            self.btn_camera.config(text="⏹  Tắt Camera")
            self._reset_scan_state()
            self._camera_loop()
        else:
            self.camera_running = False
            self.btn_camera.config(text="🎥  Bật Camera")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._reset_scan_state()
            self._set_placeholder_image()
            self.result_var.set("Chưa có kết quả")

    def _reset_scan_state(self):
        """Đặt lại trạng thái để bắt đầu ngắm/quét từ đầu."""
        self.pending_label = None
        self.pending_start_time = None
        self.is_locked = False
        self.locked_label = None
        self.locked_confidence = None
        self.btn_rescan.config(state="disabled")

    def on_rescan(self):
        """Người dùng bấm 'Quét lại' để nhận diện quả khác."""
        self._reset_scan_state()
        self.result_var.set("Đưa trái cây vào khung vuông...")

    def _get_aim_box(self, frame):
        """Tính tọa độ khung ngắm hình vuông ở giữa khung hình."""
        h, w = frame.shape[:2]
        size = int(min(h, w) * AIM_BOX_RATIO)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - size // 2, cy - size // 2
        x2, y2 = cx + size // 2, cy + size // 2
        return x1, y1, x2, y2

    def _camera_loop(self):
        if not self.camera_running or self.cap is None:
            return

        ret, raw_frame = self.cap.read()
        if not ret:
            self.result_var.set("⚠ Mất kết nối camera")
            self.root.after(100, self._camera_loop)
            return

        # Đưa khung hình về đúng kích thước cố định trước, để khung ngắm
        # luôn khớp chính xác vị trí hiển thị trên màn hình (không bị lệch).
        frame = self._fit_frame(raw_frame)
        display = frame.copy()
        x1, y1, x2, y2 = self._get_aim_box(frame)

        if self.is_locked:
            # Đã chốt kết quả -> không nhận diện tiếp, chỉ hiển thị khung xanh + kết quả
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 4)
            cv2.putText(
                display, f"DA NHAN DIEN: {self.locked_label}",
                (x1, max(y1 - 15, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2
            )
            self._show_frame_on_label(display)
            self.root.after(100, self._camera_loop)
            return

        # Chưa chốt -> tiếp tục nhận diện vùng bên trong khung ngắm
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.root.after(100, self._camera_loop)
            return

        label, confidence = self._predict(roi)
        now = time.time()

        if confidence < MIN_CONFIDENCE:
            # Không đủ tin cậy -> reset bộ đếm ổn định, chưa tính là "đang ngắm đúng"
            self.pending_label = None
            self.pending_start_time = None
            box_color = (0, 165, 255)  # cam: chưa chắc chắn
            self.result_var.set("Đưa trái cây rõ hơn vào khung vuông...")
        else:
            if label != self.pending_label:
                self.pending_label = label
                self.pending_start_time = now
            elapsed = now - self.pending_start_time
            remaining = max(0.0, STABLE_SECONDS - elapsed)

            if elapsed >= STABLE_SECONDS:
                # Đủ thời gian ổn định (STABLE_SECONDS) -> CHỐT kết quả
                self.is_locked = True
                self.locked_label = label
                self.locked_confidence = confidence
                self.result_var.set(f"✅ {label}   •   {confidence:.1f}% tin cậy")
                self.btn_rescan.config(state="normal")
                box_color = (0, 200, 0)
            else:
                box_color = (0, 255, 255)  # vàng: đang xác nhận
                self.result_var.set(
                    f"Đang xác nhận: {label} ({confidence:.0f}%) — giữ yên {remaining:.1f}s"
                )

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
        self._show_frame_on_label(display)
        self.root.after(100, self._camera_loop)

    def on_close(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
