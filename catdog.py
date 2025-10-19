# app.py
"""
Streamlit app to deploy a CNN model (cat vs dog) for prediction.
User uploads an image (jpg/png), app preprocesses it, and shows whether it's a cat or a dog.
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Streamlit App Setup ----------------
st.set_page_config(page_title="Cat vs Dog CNN Predictor", layout="centered")
st.title("🐾 Cat vs Dog — CNN Image Classifier")
st.write("Upload an image (JPG or PNG), and the model will predict whether it's a **Cat** or a **Dog**.")

# ---------------- Load Model ----------------
MODEL_NAME = "clf.h5"  # Example: "cat_dog_model.h5"


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    model = load_model(MODEL_NAME)
    return model

try:
    model = load_cnn_model()
    st.sidebar.success(f"Model '{MODEL_NAME}' loaded successfully ✅")
    st.sidebar.write(f"Model Input Shape: `{model.input_shape}`")
except Exception as e:
    st.sidebar.error(f"Error loading model '{MODEL_NAME}': {e}")
    st.stop()

# ---------------- Image Upload Section ----------------
st.header("📤 Upload Image for Prediction")


uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
else:
    st.info("Please upload a JPG or PNG image to start prediction.")


# ---------------- Preprocessing ----------------

TARGET_SIZE = (100, 100)  

def preprocess_image(pil_img: Image.Image, target_size=TARGET_SIZE):
    """Preprocess image to match model input."""
    img = pil_img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img).astype("float32") / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    return img_array

# ---------------- Prediction ----------------
def predict_image(model, img_array):
    """Predict whether the image is a cat or a dog."""
    preds = model.predict(img_array)
    # Handle binary sigmoid (1 output) or softmax (2 outputs)
    if preds.shape[1] == 1:  # Sigmoid
        prob = float(preds[0][0])
        label = "Dog" if prob >= 0.5 else "Cat"
        confidence = prob if prob >= 0.5 else 1.0 - prob
        prob_dict = {"Dog": prob, "Cat": 1.0 - prob}
    else:  # Softmax
        probs = preds[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        class_map = {0: "Cat", 1: "Dog"}
        label = class_map.get(idx, f"Class_{idx}")
        prob_dict = {class_map.get(i, f"Class_{i}"): float(probs[i]) for i in range(len(probs))}
    return label, confidence, prob_dict

# ---------------- Run Prediction ----------------
if uploaded_file is not None:
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            try:
                img_array = preprocess_image(image)
                label, confidence, prob_dict = predict_image(model, img_array)
                st.success(f"**Prediction:** {label} 🐶🐱 (Confidence: {confidence:.2%})")
                st.write("### Class Probabilities")
                for k, v in prob_dict.items():
                    st.write(f"- **{k}** : {v:.2%}")
                st.bar_chart(np.array(list(prob_dict.values())).reshape(1, -1))
                st.balloons()
            except Exception as e:
                st.error(f"Prediction failed: {e}")