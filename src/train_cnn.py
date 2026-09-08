import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import (Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D)
from tensorflow_addons.optimizers import AdamW
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy

num_classes = 7
img_rows, img_cols = 48, 48
batch_size = 16

train_data_dir = r'C:\XLAS\DoAn\XLAS_Project\dataset_new\train'
validation_data_dir = r'C:\XLAS\DoAn\XLAS_Project\dataset_new\valid'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

print("Train classes:", train_generator.class_indices)
print("Valid classes:", validation_generator.class_indices)


model = Sequential()

# Block-1
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


# Block-2
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


# Block-3
model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


# Block-4
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


# Block-5
model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Block-6
model.add(Dense(128, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Block-7
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))


model.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-5),
    metrics=['accuracy']
)

print(model.summary())


checkpoint = ModelCheckpoint(
    r'C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_delta=0.0001,
    verbose=1
)

callbacks = [checkpoint, earlystop, reduce_lr]

epochs = 60

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)