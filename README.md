# Real-Time User Recognition 🚀  

## 📌 Project Overview  
This project is a **real-time facial recognition system** that identifies users from a live video stream. It uses a **deep learning model** built with **TensorFlow** and **OpenCV**, running inside a **Docker container** for easy deployment.  

## 🛠️ Features  
- **Real-time user recognition** from a webcam feed 🎥  
- **Deep Learning model** trained for face classification 🤖  
- **Dockerized environment** for seamless execution 🐳  
- **Fast and efficient processing** with OpenCV & TensorFlow 

## 🛠️ Project Structure  

This system consists of **four Docker containers** working together:  

### 📌 1️⃣ **face_capture**  
🔹 Captures **1000 face images** per user and stores them in a **Docker volume**.  

### 📌 2️⃣ **image_storage** (Docker Volume)  
🔹 A **persistent storage** volume that holds the collected images and shares them across containers.  

### 📌 3️⃣ **training_model**  
🔹 Uses the **image_storage** volume to train a **CNN-based facial recognition model**.  
🔹 Saves the trained model for later use in real-time detection.  

### 📌 4️⃣ **detection**  
🔹 Loads the **trained CNN model** to **recognize users in real-time** using OpenCV.  

---

## 📂 Folder Structure  

```bash
📁 real-time-user-recognition-docker
 ┣ 📂 face_capture/        # Captures face images
 ┃ ┣ 📜 capture.py        # Python script for capturing images
 ┃ ┣ 📜 Dockerfile        # Docker setup for face_capture
```python
docker build -t face_capture .
```
```bash 
 ┣ 📂 training_model/      # CNN model training container
 ┃ ┣ 📜 train.py          # CNN training script
 ┃ ┣ 📜 Dockerfile        # Docker setup for training model
 ```python
docker build -t training_model .
```
```bash 
 ┣ 📂 detection/           # Real-time detection container
 ┃ ┣ 📜 detect.py         # Detects user in real-time
 ┃ ┣ 📜 Dockerfile        # Docker setup for detection
 ```python
docker build -t detect .
```
```bash 
 ┣ 📂image_storage/       # Docker volume for storing images
 ```python
docker volume create image_storage
```


