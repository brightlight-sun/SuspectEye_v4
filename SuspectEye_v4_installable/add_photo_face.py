# file to add faces by jpg images

import cv2
import face_recognition
import pickle
import os
import sys

# === SETTINGS ===
image_path = "photos/akhil.jpg"  # Your image
name = "akhil"  # Name label

# === Load existing data ===
encodings = []
names = []

if os.path.exists("data/face_encodings.pkl"):
    with open("data/face_encodings.pkl", "rb") as f:
        try:
            while True:
                encodings.extend(pickle.load(f))
        except EOFError:
            pass

if os.path.exists("data/names.pkl"):
    with open("data/names.pkl", "rb") as f:
        try:
            while True:
                names.extend(pickle.load(f))
        except EOFError:
            pass

# === Read the new photo ===
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(rgb)

if len(boxes) == 0:
    print("❌ No face detected.")
    exit()

# You can process multiple faces from one image if needed
for box in boxes:
    encoding = face_recognition.face_encodings(rgb, [box])[0]
    encodings.append(encoding)
    names.append(name)

# === Save back the full updated data ===
with open("data/face_encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)

with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

print(f"✅ Added encoding for '{name}' from {image_path}")
