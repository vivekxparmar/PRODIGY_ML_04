import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

IMG_SIZE = 64

model = load_model("gesture_model.h5")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)
inv_map = {v: k for k, v in label_map.items()}

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    roi = frame[50:300, 50:300]
    cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0
    reshaped = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(reshaped)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    gesture_name = inv_map.get(predicted_index, "Unknown")
    label = f"{gesture_name} ({confidence*100:.1f}%)"
    color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)

    cv2.putText(frame, label, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()