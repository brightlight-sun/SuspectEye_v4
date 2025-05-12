# file to detect faces by opening system camera 
# also sends the snapshot of detected photos via telegram

import cv2
import face_recognition
import pickle
import numpy as np
import time
import threading
from imutils.video import VideoStream
from playsound import playsound
import requests
import os
import sys

#new
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
#

# Load DNN model
net = cv2.dnn.readNetFromCaffe(
    resource_path("models/deploy.prototxt.txt"),
    resource_path("models/res10_300x300_ssd_iter_140000.caffemodel")
)
 
def play_beep():
    playsound(resource_path("beepSound.wav"))
    
# Load encodings
encodings, names = [], []
try:
    with open(resource_path("data/face_encodings.pkl"), "rb") as ef:
        while True:
            try:
                encodings.extend(pickle.load(ef))
            except EOFError:
                break

    with open(resource_path("data/names.pkl"), "rb") as nf:
        while True:
            try:
                names.extend(pickle.load(nf))
            except EOFError:
                break
except FileNotFoundError:
    print("No encoded faces found.")
    exit()

# Telegram config
BOT_TOKEN = 'your_telegram_bot_token'
CHAT_ID = 'your_telegram_chat_id'

# Create temp directory if not exists
os.makedirs("temp", exist_ok=True)

# Telegram send function
def notify_telegram_with_photo(name, image):
    try:
        # not needed, now sending snapshots
        # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # cv2.putText(image, timestamp, (10, image.shape[0] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        filename = f"temp/{name}_{int(time.time())}.jpg"
        cv2.imwrite(filename, image)

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(filename, 'rb') as photo:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": f"🟢 {name} detected by SuspectEye"}, files={"photo": photo})
    except Exception as e:
        print(f"[Telegram Error] {e}")

# Beep config
last_known_name = None
cooldown_time = 3
last_beep_time = 0

# Cooldown for group photo
last_photo_times = {}
PHOTO_COOLDOWN = 10

# Start camera
vs = VideoStream(src=0).start()
time.sleep(2.0)
start_time = time.time()
frame_count = 0

while True:
    frame = vs.read()
    if frame is None:
        print("⚠️ Failed to read frame.")
        continue

    frame_count += 1

    frame = cv2.resize(frame, (600, 500))
    (h, w) = frame.shape[:2]

    # Display overlays
    elapsed_time = int(time.time() - start_time)
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    date_text = time.strftime("Date: %A, %Y-%m-%d")
    time_text = time.strftime("Time: %H:%M:%S")

    #overlay background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 75), (0, 0, 0), -1)
    cv2.rectangle(overlay, (5, h - 40), (200, h - 5), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
   
    # draw date and time
    cv2.putText(frame, date_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"Timer: {formatted_time}", (10, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)

    # DNN Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    frame_marked = frame.copy()
    known_faces_detected = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)

            if not face_encodings:
                print("⚠️ No face encoding found.")
                continue

            encoding = face_encodings[0]
            distances = face_recognition.face_distance(encodings, encoding)
            min_dist = np.min(distances)
            index = np.argmin(distances)
            threshold = 0.5

            if min_dist < threshold:
                name = names[index]
                color = (0, 255, 0)
                known_faces_detected.append(name)
                
                if name != last_known_name or (time.time() -    last_beep_time) > cooldown_time:
                    threading.Thread(target=play_beep).start()
                    last_known_name = name
                    last_beep_time = time.time()
            else:
                name = "Unknown"
                color = (0, 0, 255)
                last_known_name = None

            print(f"Distance: {min_dist:.2f} | Prediction: {name}")

            cv2.rectangle(frame_marked, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame_marked, name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

            

    # Send group image if known face found
    if known_faces_detected:
        current_time = time.time()
        last_time = last_photo_times.get("group", 0)
        if current_time - last_time > PHOTO_COOLDOWN:
            threading.Thread(target=notify_telegram_with_photo, args=("group", frame_marked)).start()
            last_photo_times["group"] = current_time

    cv2.imshow("SuspectEye v4", frame_marked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
