# Face Recognition using OpenCV & Machine Learning

## Overview

This project is a **real-time face recognition system** built using **OpenCV** and **Local Binary Patterns Histograms (LBPH)**. It captures facial images, trains a model, and recognizes faces in real-time using a webcam. The project is structured into three key modules:

1. **Face Data Collection** – Capturing and storing face images.
2. **Model Training** – Training the model using LBPH Face Recognizer.
3. **Face Recognition** – Detecting and recognizing faces in real-time.

## Features

- **Face Detection** using Haar Cascade Classifier.
- **Face Recognition** using LBPH algorithm.
- **Real-time video processing** with OpenCV.
- **Interactive Command-line Interface** for easy use.
- **Modular and Scalable Code** for future enhancements.
- **Trained model storage** using OpenCV and NumPy.

## Installation

### Prerequisites

Ensure you have Python installed (Python 3.6+ recommended). Install the required dependencies using:

```bash
pip install opencv-contrib-python numpy
```

## How to Use

Run the script and choose an option:

```bash
python script.py
```

1. **Capture Faces** - Enter `1` and provide a name to collect face data.
2. **Train Model** - Enter `2` to train the model on collected face data.
3. **Recognize Faces** - Enter `3` to recognize faces in real-time using the webcam.

## Project Structure

```
Face-Recognition-Project/
│-- dataset/               # Stores captured face images
│-- face_recognizer.yml    # Trained face recognition model
│-- names.npy              # Stores person labels
│-- script.py              # Main script for face recognition
│-- README.md              # Project documentation
```

## How It Works

### 1. Face Data Collection

- The script captures **100 grayscale images** of a person’s face.
- Faces are detected using **Haar Cascade Classifier** and stored in the `dataset/` folder under the person's name.
- The images are resized and saved for training.

### 2. Model Training

- The system loads all images from the `dataset/` folder.
- Each person is assigned a unique label.
- The **LBPH Face Recognizer** is trained on the dataset.
- The trained model is saved as `face_recognizer.yml`, and label names are stored in `names.npy`.

### 3. Face Recognition

- The trained model is loaded.
- The webcam feed is processed in real-time.
- Faces are detected and **predicted** using the trained model.
- The predicted **name and confidence score** are displayed on the screen.

## Technologies Used

- **OpenCV** – Image processing and face detection.
- **NumPy** – Data handling and label management.
- **LBPH Face Recognizer** – Face recognition algorithm.

## Future Enhancements

- Implement **Deep Learning-based** face recognition with DNN models.
- Improve accuracy by incorporating **preprocessing techniques**.
- Add a **GUI-based interface** for ease of use.
- Deploy the model as a **web application** using Flask or FastAPI.

---

## Author

Developed by ASHWANI SONI
