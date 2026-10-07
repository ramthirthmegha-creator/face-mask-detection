# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------- SETTINGS --------
DATA_DIR = "dataset"          # must contain: dataset/with_mask and dataset/without_mask
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# -------- DATA --------
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.20  # 20% for validation
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# -------- MODEL --------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')   # binary: mask / no-mask
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------- TRAIN --------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------- SAVE --------
model.save("mask_detector.h5")
print("✅ Model trained and saved as mask_detector.h5")