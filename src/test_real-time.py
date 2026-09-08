import cv2
import numpy as np
from keras.models import load_model
import tensorflow_addons as tfa

model = load_model(r"C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5")
face_detect = cv2.CascadeClassifier(r"C:\XLAS\DoAn\XLAS_Project\haarcascade_frontalface_default.xml")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48,48))
        roi_normalized = roi_resized.reshape(1,48,48,1) / 255.0

        pred = model.predict(roi_normalized)
        emotion = emotion_labels[np.argmax(pred)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
