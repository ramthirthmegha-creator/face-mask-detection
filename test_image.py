import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# load model and detector
model = load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# read image
img = cv2.imread("group.jpg")      # replace with your file name
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (128,128))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0][0]
    label = "Mask" if pred < 0.5 else "No Mask"
    color = (0,255,0) if label == "Mask" else (0,0,255)

    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(img, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Photo Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()