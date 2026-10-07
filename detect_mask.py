import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detector.h5")

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128, 128))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        pred = model.predict(face)[0][0]

        # Print the prediction value (for debugging)
        print(f"Prediction value: {pred:.3f}")

        # Adjust threshold slightly for better results
        if pred > 0.5:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Mask"
            color = (0, 255, 0)

        # Display rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({pred:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()