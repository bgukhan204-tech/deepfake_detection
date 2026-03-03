import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

import os
import gdown

st.set_page_config(page_title="Deepfake Detection", layout="centered")

st.title("🕵️ Deepfake Image Detection")
st.write("Upload an image to check if it is REAL or MANIPULATED")

model_path = "model/deepfake_model.h5"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1PRQ2PNMJKJhJPYWkeAWpjksr6A26hSPF"
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error(f"❌ MANIPULATED IMAGE\nConfidence: {prediction*100:.2f}%")
    else:
        st.success(f"✅ REAL IMAGE\nConfidence: {(1-prediction)*100:.2f}%")

prediction = model.predict(img)[0][0]
confidence = float(prediction)

if prediction > 0.5:
    final_conf = confidence
    st.error("❌ MANIPULATED IMAGE")
else:
    final_conf = 1 - confidence
    st.success("✅ REAL IMAGE")

st.write(f"### Confidence: {final_conf*100:.2f}%")
st.progress(final_conf)