import cv2
import numpy as np
from keras.models import load_model
import tensorflow_addons as tfa

model = load_model(r'C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5')
face_detect = cv2.CascadeClassifier(r'C:\XLAS\DoAn\XLAS_Project\haarcascade_frontalface_default.xml')

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

frame = cv2.imread(r"C:\XLAS\DoAn\XLAS_Project\emotion_.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi_normalized = roi_resized.reshape(1, 48, 48, 1) / 255.0
    pred = model.predict(roi_normalized, verbose=1) 
    emotion = emotion_labels[np.argmax(pred)]
    
    print(f"Emotion: {emotion}")

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1) 
    cv2.putText(frame, 
                emotion, 
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8,
                (255, 255, 255),
                2)

window_name = 'Emotion Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

cv2.imshow(window_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()