# 🏐 ReflectAI: Aquaball Line Call Detection with YOLOv8

## 📌 Project Overview

This project uses computer vision and machine learning to **detect, track, and make line calls** for a game of Aquaball (volleyball in a pool) using **YOLOv8** and **OpenCV**. The goal is to automate the process of determining whether the ball landed "in" or "out" during gameplay.

---

## 🔧 Technologies Used

- **Python** – Primary programming language.
- **YOLO (You Only Look Once)** – Neural network model for real-time object detection.
- **OpenCV** – Handles image/video processing, drawing, and perspective transformation.
- **Numpy** – Efficient numerical computation and matrix operations.
- **Porcupine + PyAudio** – Voice activation engine used specifically to detect when a player says a challenge keyword to trigger a slow-motion replay.
- **LabelImg** – Image annotation tool used for creating training datasets.
- **TensorFlow / PyTorch** – Deep learning frameworks for model training and evaluation.
- **Tensors** – Multidimensional arrays used in neural network data flow.
- **Perspective Transformation** – Converts camera perspective to a fixed, bird’s-eye view for better spatial analysis.
- **Object Tracking** – Maintains object consistency across frames.
- **Real-Time Processing** – Optimized for low-latency visual analysis.

---

## ⚙️ System Architecture

1. **Live Camera Feed**  
   Captured and processed using OpenCV.

2. **Object Detection**  
   YOLO runs on each frame, providing bounding boxes and classifications.

3. **Object Tracking**  
   Tracks identities across frames to maintain consistent object information.

4. **Voice Command for Challenge**  
   Porcupine listens for a specific wake word (e.g., "challenge") via PyAudio.
   When triggered, a slow-motion replay of the previous few seconds is displayed.

5. **Line Call Logic**  
   Determine whether the ball landed in or out based on its position relative to the court.

---

## 🏗️ How It Was Built

### 1. 📸 Data Collection & Annotation
- Images were collected from relevant scenarios for training.
- **LabelImg** was used to annotate images with bounding boxes.
- Annotations were exported in **YOLO format** to be compatible with the training pipeline.

### 2. 🧠 Model Training
- A custom **YOLO** model was trained using **PyTorch**.
- Training involved feeding annotated data as **tensors** into the neural network.
- The model was optimized through multiple sets of training for real-time object detection accuracy and performance.

### 3. ⚙️ Real-Time Integration
- Live video feed is processed using **DroidCam**.
- Each frame is passed through the YOLO model for object detection.
- Detected objects are tracked using an object tracking module to maintain continuity.
- **PyAudio** captures audio input and feeds it to **Porcupine** for wake word detection.
- When the keyword (e.g., "challenge") is detected, the system triggers a **slow-motion replay** of the most recent video segment.
- **Perspective transformation** is applied where necessary to adjust the visual orientation of the feed.

---

## 📚 Citation
```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}