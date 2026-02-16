Person Identity & Attribute Analysis System

A modular deep learning–based computer vision application for real-time face detection, recognition, age prediction, gender classification, emotion analysis, and weapon detection (in progress).

📸 Demo Screenshot

Real-time face detection with emotion, gender, and age prediction overlay.

🚀 Features

Real-time Face Detection & Recognition using InsightFace (ArcFace embeddings)

Age Prediction using MobileNetV3 (PyTorch)

Gender Classification using UTKFace-trained CNN model

Emotion Recognition using CNN-based FER model

Desktop GUI built with PyQt

CUDA-enabled GPU acceleration

Checkpoint-based training with resume capability

Gun Detection Module (YOLOv8) – Under Development

🧠 Models Used
Module	Model
Face Detection & Recognition	InsightFace (ArcFace)
Age Prediction	MobileNetV3 (.pth)
Gender Classification	UTKFace CNN (.h5)
Emotion Recognition	CNN FER Model (.keras)
Gun Detection (WIP)	YOLOv8
🏗 System Architecture
Input (Webcam / Video / Image)
        │
        ▼
Face Detection (InsightFace)
        │
        ├───────────────┬───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
   Face Recognition    Age          Gender         Emotion
        │
        ▼
   Annotated Output (PyQt GUI)

🛠 Tech Stack

Python

PyTorch

TensorFlow / Keras

InsightFace

OpenCV

PyQt

CUDA

📦 Installation
git clone https://github.com/yourusername/person-identity-app.git
cd person-identity-app
pip install -r requirements.txt

▶️ Run the Application
python main.py

📊 Applications

Smart Surveillance Systems

AI-based Monitoring

Behavioral Analytics

IoT Alert Integration

🔮 Future Improvements

Multi-person tracking with persistent IDs

Full Gun Detection integration

Edge device optimization

Cloud-based API deployment
