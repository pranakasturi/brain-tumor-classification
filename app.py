import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor Classification from MRI")

# Load models
@st.cache_resource
def load_models():
    custom_model = load_model("models/custom_cnn_best.h5")
    mobilenet_model = load_model("models/mobilenetv2_best.h5")
    return custom_model, mobilenet_model

custom_model, mobilenet_model = load_models()

# Class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Sidebar - model selection
model_choice = st.sidebar.radio("Choose Model", ("Custom CNN", "MobileNetV2"))

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if model_choice == "Custom CNN":
        pred = custom_model.predict(img_array)
    else:
        pred = mobilenet_model.predict(img_array)

    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Display result
    st.markdown("### 🩺 Prediction Result")
    st.success(f"**{predicted_class}** (Confidence: {confidence:.2f}%)")
else:
    st.info("Please upload an MRI image to begin classification.")
