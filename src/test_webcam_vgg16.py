import cv2
import numpy as np
import tensorflow as tf

# ==============================
# 1. LOAD MODEL
# ==============================
model = tf.keras.models.load_model(
    r"C:\XLAS\DoAn\XLAS_Project\Emotion_VGG16_Optimized.h5"
)

emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# ==============================
# 2. LOAD FACE CASCADE
# ==============================
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\PC\Downloads\XLAS_Project-main_2\XLAS_Project-main_2\haarcascade_frontalface_default.xml"
)

# ==============================
# 3. OPEN WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

print("Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # ------------------------------
        # 4. PREPROCESS (GIỐNG TRAIN)
        # ------------------------------
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)   # (1,48,48,3)

        # ------------------------------
        # 5. PREDICT
        # ------------------------------
        preds = model.predict(face, verbose=0)
        emotion_id = np.argmax(preds)
        emotion = emotion_labels[emotion_id]
        confidence = preds[0][emotion_id] * 100

        # ------------------------------
        # 6. DISPLAY
        # ------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Recognition - VGG16", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
