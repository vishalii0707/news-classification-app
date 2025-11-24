import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import json

# ================================
# STEP 1 — Dataset Path
# ================================
data_path = "news_images/"   # your dataset folder

# ================================
# STEP 2 — Preprocessing & Augmentation
# ================================
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ================================
# STEP 3 — Build MobileNetV2 Model
# ================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze pretrained weights

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# STEP 4 — Train the Model
# ================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ================================
# STEP 5 — Plot Accuracy & Loss
# ================================
plt.figure(figsize=(12, 5))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.legend()

plt.show()

# ================================
# STEP 6 — Save the Model
# ================================
model.save("news_classifier_model.h5")
print("Model saved as news_classifier_model.h5")

# ================================
# SAVE ACCURACY FOR STREAMLIT APP
# ================================
final_acc = history.history['accuracy'][-1]

with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(final_acc)}, f)

print("Accuracy saved to metrics.json")


# ================================
# STEP 7 — Predict on a Single Image
# ================================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Mapping index → label
    labels = list(train_data.class_indices.keys())
    print("Predicted Category:", labels[class_index])

# Example:
# predict_image("test.jpg")

