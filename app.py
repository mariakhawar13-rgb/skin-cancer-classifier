import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
from gradcam import make_gradcam_heatmap, overlay_heatmap
from helper import preprocess_image, load_class_names

st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")
st.title("🔬 Skin Cancer Classifier + Grad-CAM")

# ---------------------------
# Load model (.h5)
# ---------------------------
@st.cache_resource
def load_model(path="skin_cancer_model.h5"):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}. Upload it to the repo root or update path.")
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error("Failed to load model: " + str(e))
        return None

model = load_model("skin_cancer_model.h5")
CLASS_NAMES = load_class_names()

# ---------------------------
# Input UI
# ---------------------------
col1, col2 = st.columns([1,1])
with col1:
    st.header("Input")
    upload = st.file_uploader("Upload image (jpg, png)", type=["jpg","jpeg","png"])
    camera = st.camera_input("Or take a photo (camera)")
with col2:
    st.header("Prediction & Grad-CAM")
    placeholder = st.empty()

image_obj = None
if upload:
    image_obj = Image.open(upload).convert("RGB")
elif camera:
    image_obj = Image.open(camera).convert("RGB")

if image_obj is not None:
    st.image(image_obj, caption="Original image", use_column_width=True)
    # preprocess for model (helper returns uint8 array)
    img_array = preprocess_image(image_obj)            # shape (224,224,3) uint8
    # model expects preprocessed (EfficientNet) inside model (we included preprocess_input in model),
    # so we need to expand dims and pass raw pixel range [0,255] (model's internal preprocess will run)
    input_batch = np.expand_dims(img_array.astype("float32"), axis=0)

    if model is None:
        st.error("Model not loaded. Please check file.")
    else:
        preds = model.predict(input_batch, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx])
        st.subheader(f"Prediction: {CLASS_NAMES[idx].title()} ({conf*100:.2f}%)")

        # Grad-CAM
        heatmap, pred_idx = make_gradcam_heatmap(model, input_batch)
        overlay = overlay_heatmap(img_array, heatmap)

        # show heatmap + overlay
        st.image(overlay, caption=f"Grad-CAM (predicted: {CLASS_NAMES[pred_idx].title()})", use_column_width=True)

st.markdown("---")
st.info("This is a research/demo tool — not a diagnostic device.")
