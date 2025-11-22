import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---- SETTINGS ----
model_file = "emotion_model_fast.h5"  # your trained model
train_dir = "train_fast"              # folder to auto-read class labels
img_size = (48, 48)
# -------------------

# Load model
model = load_model(model_file)
print(f"✅ Loaded model: {model_file}")

# Automatically get emotion labels from train folder
emotion_labels = sorted(os.listdir(train_dir))
print("Detected emotion labels:", emotion_labels)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

print("Webcam started. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        if roi_gray.size == 0:
            continue
        roi = cv2.resize(roi_gray, img_size)
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        preds = model.predict(roi)
        label = emotion_labels[int(np.argmax(preds))]
        score = float(np.max(preds))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection Fast", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




