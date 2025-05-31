# Real-Time Hand Gesture Recognition

## Overview

This project implements a real-time hand gesture recognition system using a Convolutional Neural Network (CNN) and OpenCV. The model is trained on the LeapGestRecog dataset and deployed to recognize gestures live from webcam input.

## Dataset

- **Name:** LeapGestRecog
- **Structure:** The dataset contains multiple folders, each corresponding to a subject. Each subject folder contains gesture subfolders like `00_palm`, `01_l`, etc.
- **Input Type:** Grayscale images
- **Image Size:** 64x64 (resized during preprocessing)

## Workflow

1. **Mount Google Drive (Colab)**  
   Load the dataset stored on Google Drive for training.

2. **Data Preprocessing**  
   - Load all grayscale gesture images
   - Resize images to 64x64
   - Normalize pixel values
   - Encode gesture labels into one-hot vectors

3. **Model Architecture**  
   - Conv2D -> BatchNorm -> MaxPooling
   - Conv2D -> BatchNorm -> MaxPooling
   - Flatten -> Dense -> Dropout -> Output

4. **Training**  
   - Model trained for 15 epochs with `categorical_crossentropy` loss and `adam` optimizer
   - Achieved high validation accuracy

5. **Saving Model**  
   - Model saved as `gesture_model.h5`
   - Gesture label mapping saved as `label_map.pkl`

6. **Real-Time Prediction**  
   - Live webcam input using OpenCV
   - Region of Interest (ROI) defined for gesture input
   - CNN model predicts gesture class and displays the result with confidence score

## Requirements

- Python
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- pickle

## How to Run

### 1. Download the zip file
### 2. Open the folder in VS Code
### 3. Run the .py File
