import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Load trained model
MODEL_PATH = "/models/model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels (usernames)
DATASET_PATH = "/image_storage"
users = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (128, 128))  # Adjust to match your model's expected input size
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Predict the user
    prediction = model.predict(img)
    user_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display result
    label = f"{users[user_index]} ({confidence*100:.2f}%)"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

# Release resources
camera.release()
plt.close()
