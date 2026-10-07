# 😷 Face Mask Detection System

A Computer Vision and Machine Learning project that detects whether a person is wearing a face mask or not. The system uses OpenCV for face detection and a trained machine learning model for mask classification.

## 📌 Project Overview

The Face Mask Detection System is designed to identify faces and determine whether a person is wearing a mask.

The project includes functionality for:

- Loading and preparing the dataset
- Training the mask detection model
- Detecting faces using Haar Cascade
- Testing mask detection on images
- Detecting masks in real-time using a webcam

## ✨ Features

- 😷 Detects **Mask** and **No Mask**
- 👤 Detects faces using OpenCV Haar Cascade
- 📷 Supports image-based testing
- 🎥 Supports real-time webcam detection
- 🧠 Uses a trained machine learning model
- ⚡ Provides real-time detection results

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Machine Learning**
- **Haar Cascade Classifier**

## 📂 Project Structure

```text
Face-Mask-Detection/
│
├── detect_mask.py
├── haarcascade_frontalface_default.xml
├── load_dataset.py
├── test_camera.py
├── test_image.py
├── train_model.py
└── README.md

## ⚙️ How It Works

The system follows these basic steps:

```text
Camera / Image
      ↓
Face Detection
      ↓
Extract Face Region
      ↓
Mask Classification
      ↓
Display Result
```

1. The system captures frames from the camera.
2. OpenCV processes each frame.
3. Faces are detected within the frame.
4. The detected face region is passed to the mask classification model.
5. The model predicts whether the person is wearing a mask.
6. The prediction is displayed on the video feed.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Face-Mask-Detection.git
```

### 2. Navigate to the Project Folder

```bash
cd Face-Mask-Detection
```

### 3. Install Required Libraries

```bash
pip install opencv-python numpy
```

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python test_camera.py
```

Make sure your computer has a working webcam if your implementation uses real-time camera detection.


## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Computer Vision concepts
- Working with OpenCV
- Face detection and image processing
- Working with image datasets
- Implementing machine learning-based classification
- Processing real-time webcam input
- Understanding how AI can be applied to real-world problems

## 🔮 Future Improvements

Possible improvements include:

- Improve detection accuracy
- Detect multiple faces simultaneously
- Add confidence scores
- Add support for different types of masks
- Improve performance in different lighting conditions
- Deploy the system as a web application
- Add real-time alerts for people without masks

## 👩‍💻 Author

**Megha Ramthirth**

B.E. Computer Science Engineering (Data Science)

---

⭐ If you found this project useful, consider giving the repository a star!
