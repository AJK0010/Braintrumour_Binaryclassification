import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Brain Tumour Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumour Binary Classification")
st.write("Upload an MRI image to predict whether it is **Tumour** or **No Tumour**.")

# =========================
# LOAD MODEL (SAFE PATH)
# =========================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor.h5")
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model file not found. Please ensure brain_tumor.h5 is in the same folder.")
    st.stop()

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(processed_image)[0][0]

    if prediction >= 0.5:
        st.success(f"🧠 Tumour Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Tumour Detected (Confidence: {1 - prediction:.2f})")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Developed for Brain Tumour Binary Classification Project")
