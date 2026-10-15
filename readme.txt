# 🎯 Real-Time Object Detection with Python

A real-time object detection project using **Python, OpenCV, and YOLOv8**. This application uses a webcam to detect objects in live video, draw bounding boxes around them, and display their labels and confidence scores in Persian.

## ✨ Features

- 📷 Real-time object detection using a webcam
- 🤖 YOLOv8 object detection model
- 🟩 Bounding boxes around detected objects
- 🇮🇷 Persian labels for supported object classes
- 📊 Confidence scores displayed as percentages
- ⚡ Adjustable camera resolution and detection confidence
- ⌨️ Press `Q` to exit the application

## 🛠️ Technologies Used

- **Python** — Programming language
- **OpenCV** — Webcam access, image processing, and visualization
- **Ultralytics YOLOv8** — Object detection model
- **PyTorch** — Deep learning framework used by YOLO

## 📦 Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install opencv-python ultralytics
```

**Requirements:** Use a Python version supported by your installed Ultralytics and PyTorch versions. Python 3.7 is too old for many current releases.

## 🚀 Usage

Run the application:

```bash
python object_detector.py
```

On the first run, the YOLOv8 model file (`yolov8n.pt`) may be downloaded automatically.

Make sure your webcam is connected and accessible.

## 🧠 How It Works

1. OpenCV captures frames from the webcam.
2. YOLOv8 analyzes each frame and detects objects.
3. The program retrieves the detected classes and confidence scores.
4. Bounding boxes are drawn around the detected objects.
5. Supported English class names are converted to Persian labels.
6. The processed video is displayed in a window.

## 🏷️ Supported Objects

The application can detect various objects supported by the pretrained YOLOv8 model, including:

- Person
- Cell phone
- Book
- Laptop
- Keyboard and mouse
- Bottle and cup
- Chair and backpack
- Clock and television
- Dog, cat, and bird
- Car, bicycle, bus, and truck
- Fruits and other everyday objects

The exact objects detected depend on the classes included in the pretrained model.

## ⚙️ Configuration

You can adjust the detection settings in the source code:

```python
results = model.predict(
    source=frame,
    imgsz=640,
    conf=0.35,
    verbose=False
)
```

- `imgsz=640`: Image size used for inference.
- `conf=0.35`: Minimum confidence threshold for detections.
- `verbose=False`: Disables detailed prediction logs.

## ⚠️ Notes

- A webcam is required for live detection.
- Detection speed depends on your computer's hardware.
- The pretrained model does not recognize every possible object.
- OpenCV's default text rendering may not display Persian characters correctly. Pillow can be used to render Persian text properly.
- This project uses a pretrained model and does not train a new model.

## 📄 License

Check the licenses of the libraries and pretrained model before redistributing the project or its model weights.

---

**Built with Python, OpenCV, and YOLOv8.**