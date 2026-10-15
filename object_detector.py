
import cv2
from ultralytics import YOLO

# بارگذاری مدل تشخیص اشیاء
# اجرای اول ممکن است مدل را دانلود کند.
model = YOLO("yolov8n.pt")

# نام کلاس‌های مدل به فارسی
persian_labels = {
    "person": "شخص",
    "cell phone": "موبایل",
    "book": "کتاب",
    "laptop": "لپ‌تاپ",
    "keyboard": "کیبورد",
    "mouse": "ماوس",
    "bottle": "بطری",
    "cup": "لیوان",
    "chair": "صندلی",
    "backpack": "کوله‌پشتی",
    "handbag": "کیف دستی",
    "suitcase": "چمدان",
    "tv": "تلویزیون",
    "remote": "کنترل",
    "clock": "ساعت",
    "scissors": "قیچی",
    "pen": "خودکار",
    "banana": "موز",
    "apple": "سیب",
    "orange": "پرتقال",
    "pizza": "پیتزا",
    "bowl": "کاسه",
    "fork": "چنگال",
    "knife": "چاقو",
    "spoon": "قاشق",
    "vase": "گلدان",
    "potted plant": "گلدان گیاه",
    "sports ball": "توپ",
    "umbrella": "چتر",
    "tie": "کراوات",
    "dog": "سگ",
    "cat": "گربه",
    "bird": "پرنده",
    "horse": "اسب",
    "car": "خودرو",
    "motorcycle": "موتورسیکلت",
    "bicycle": "دوچرخه",
    "bus": "اتوبوس",
    "truck": "کامیون",
    "train": "قطار",
    "traffic light": "چراغ راهنمایی",
    "person": "شخص",
}

# باز کردن دوربین لپ‌تاپ
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("خطا: دوربین لپ‌تاپ باز نشد.")
    raise SystemExit

# تنظیم وضوح تصویر
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("تشخیص اشیاء شروع شد.")
print("برای خروج، کلید Q را فشار بده.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("دریافت تصویر از دوربین ناموفق بود.")
            break

        # اجرای مدل روی تصویر زنده
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        # ترسیم کادرها و برچسب‌ها
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                english_name = model.names[class_id]
                persian_name = persian_labels.get(
                    english_name,
                    english_name
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # کادر دور شیء
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 80),
                    2
                )

                # متن برچسب
                label = f"{persian_name} {confidence:.0%}"

                # پس‌زمینه برچسب
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    2
                )

                label_y = max(y1, text_h + 10)

                cv2.rectangle(
                    frame,
                    (x1, label_y - text_h - 8),
                    (x1 + text_w + 8, label_y + baseline),
                    (0, 150, 55),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # نمایش تصویر دوربین
        cv2.imshow(
            "Real-Time Object Detection - Q to Quit",
            frame
        )

        # خروج با Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
 