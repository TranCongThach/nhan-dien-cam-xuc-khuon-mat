import json
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_addons as tfa

# ===============================
# PATHS
# ===============================
model_path = r"C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5"
test_data_dir = r"C:\XLAS\DoAn\XLAS_Project\dataset_new\test"

save_test_history = r"C:\XLAS\DoAn\XLAS_Project\history_test.json"
save_confusion_matrix = r"C:\XLAS\DoAn\XLAS_Project\confusion_matrix.txt"
save_classification_report = r"C:\XLAS\DoAn\XLAS_Project\classification_report.txt"

# ===============================
# TEST DATA LOADING
# ===============================
img_rows, img_cols = 48, 48
batch_size = 16

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_directory(
    test_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())
print("Class mapping:", test_generator.class_indices)

# ===============================
# LOAD MODEL
# ===============================
print("\nLoading model…")
model = load_model(model_path)
print("Loaded:", model_path)

# ===============================
# MODEL EVALUATION
# ===============================
print("\nEvaluating on TEST set…")
test_loss, test_acc = model.evaluate(test_generator)

print("\n============================")
print("  TEST LOSS     :", test_loss)
print("  TEST ACCURACY :", test_acc)
print("============================")

# ===============================
# SAVE TEST METRICS
# ===============================
test_history = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc)
}

with open(save_test_history, "w") as f:
    json.dump(test_history, f, indent=4)

print("Saved test history →", save_test_history)

# ===============================
# PREDICT ON TEST SET
# ===============================
print("\nPredicting labels…")
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)

true_classes = test_generator.classes

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(true_classes, pred_classes)
np.savetxt(save_confusion_matrix, cm, fmt='%d')

print("Saved confusion matrix →", save_confusion_matrix)

# ===============================
# CLASSIFICATION REPORT
# ===============================
report = classification_report(true_classes, pred_classes, target_names=class_labels)

with open(save_classification_report, "w") as f:
    f.write(report)

print("Saved classification report →", save_classification_report)

print("\n=== TEST DONE ✓ ===")