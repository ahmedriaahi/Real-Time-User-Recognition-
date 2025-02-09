import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle  # Ensure dataset is shuffled

# Define paths
dataset_path = "/image_storage"  # Adjusted path to user folders
model_save_path = "/models/model.keras"

# Load dataset
def load_dataset(dataset_path, img_size=(128, 128)):
    X, y = [], []
    label_map = {}  # Map user names to labels
    label_count = 0

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset path not found: {dataset_path}")

    for user_id in sorted(os.listdir(dataset_path)):  # Sorted for consistency
        user_folder = os.path.join(dataset_path, user_id)
        if os.path.isdir(user_folder):
            if user_id not in label_map:
                label_map[user_id] = label_count
                label_count += 1
            
            for img_name in os.listdir(user_folder):
                img_path = os.path.join(user_folder, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"⚠️ Skipping unreadable file: {img_path}")
                    continue
                
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure 3-channel RGB
                X.append(img)
                y.append(label_map[user_id])

    if not X:
        raise ValueError("❌ No images found in dataset!")

    return np.array(X), np.array(y), label_map

# Load and preprocess dataset
X, y, label_map = load_dataset(dataset_path)
X = X / 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=len(label_map))  # Convert labels
X, y = shuffle(X, y, random_state=42)  # Shuffle dataset

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_map), activation='softmax')  # Multi-user classification
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print(f"📢 Training model with {len(X)} images and {len(label_map)} users...")
model.fit(X, y, epochs=8, batch_size=16, validation_split=0.2)

# Save model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
model.save(model_save_path)
print(f"✅ Model saved at {model_save_path}")

