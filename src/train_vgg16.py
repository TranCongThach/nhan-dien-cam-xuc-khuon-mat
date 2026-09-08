import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# CONFIG
num_classes = 7
img_rows, img_cols = 48, 48
batch_size = 64      # Tăng batch để training ổn định hơn

train_data_dir = r'C:\Users\PC\Downloads\XLAS_Project-main_2\XLAS_Project-main_2\dataset_new\train'
validation_data_dir = r'C:\Users\PC\Downloads\XLAS_Project-main_2\XLAS_Project-main_2\dataset_new\valid'


# ================================
# 1. DATA AUGMENTATION TỐT NHẤT
# ================================
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255.)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


# ================================
# 2. LOAD VGG16 — FINE TUNE LAYER
# ================================
base_model = VGG16(include_top=False, weights="imagenet",
                   input_shape=(img_rows, img_cols, 3))

# Freeze 10 layer đầu → fine-tune phần còn lại
for layer in base_model.layers[:10]:
    layer.trainable = False
for layer in base_model.layers[10:]:
    layer.trainable = True


# ================================
# 3. TOP LAYERS
# ================================
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-5),   # LR rất nhỏ để fine-tune
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ================================
# 4. CALLBACKS
# ================================
checkpoint = ModelCheckpoint(
    r'C:\XLAS\DoAn\XLAS_Project\Emotion_VGG16_Optimized.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, earlystop, reduce_lr]


# ================================
# 5. TRAIN
# ================================
epochs = 50   # cần nhiều epoch để fine-tune

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)
# ===================================
# 6. VẼ BIỂU ĐỒ TRAIN / VALIDATION
# ===================================
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

# -------------------------------
# Biểu đồ Accuracy
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, 'b', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy (VGG16)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Biểu đồ Loss
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, 'b', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss (VGG16)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
