import cv2
import os

# Ask for the user's name
user_name = input("Enter user name: ").strip()

# Define storage path for this user inside the volume
SAVE_PATH = f"/image_storage/{user_name}"
os.makedirs(SAVE_PATH, exist_ok=True)

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

# Capture 1000 images
for i in range(1000):  # Changed from 100 to 1000
    ret, frame = camera.read()
    if not ret:
        print(f"Failed to capture image {i}")
        continue
    filename = os.path.join(SAVE_PATH, f"{user_name}_image_{i+1}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

# Release the camera
camera.release()

print(f"1000 images saved for user: {user_name}")

