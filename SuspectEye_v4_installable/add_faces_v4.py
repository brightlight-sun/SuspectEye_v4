# file to add faces using systems camera

import cv2
import face_recognition
import pickle
import os

os.makedirs("data", exist_ok=True)

import sys
name = sys.argv[1] if len(sys.argv) > 1 else "unknown"


video = cv2.VideoCapture(0)
count = 0
embeddings = []
names = []

while True:
    ret, frame = video.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    for box in boxes:
        encoding = face_recognition.face_encodings(rgb, [box])[0]
        embeddings.append(encoding)
        names.append(name)
        count += 1

        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({count})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Save data
with open("data/face_encodings.pkl", "ab") as f:
    pickle.dump(embeddings, f)

with open("data/names.pkl", "ab") as f:
    pickle.dump(names, f)

print("[INFO] Face embeddings saved.")