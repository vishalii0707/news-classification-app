import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

model = tf.keras.models.load_model("news_classifier_model.h5")

# Load accuracy
with open("metrics.json", "r") as f:
    metrics = json.load(f)
accuracy_percent = round(metrics["accuracy"] * 100, 2)

labels = ['business', 'politics', 'sports']

st.title("📰 News Article Classification")
st.write("Upload an image and the model will predict the news category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.success(f"### 🏆 Predicted Category: **{labels[class_index].upper()}**")
    st.info(f"📊 Model Training Accuracy: **{accuracy_percent}%**")
