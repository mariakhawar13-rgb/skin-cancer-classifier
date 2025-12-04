# ✅ FIRST Streamlit command
import streamlit as st
st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🔬", layout="wide")

# Now import everything else
import numpy as np
from PIL import Image
import cv2
import os

# Import TensorFlow with specific configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import YOUR gradcam.py
from gradcam import generate_gradcam

# Class names
CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

IMG_SIZE = (224, 224)


# FIXED MODEL LOADING (.h5 version)
# -------------------------------
@st.cache_resource
def load_model():
    """Load .h5 model safely (EfficientNet-compatible)"""
    model_path = "skin_cancer_model_final.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.write("Available files:", os.listdir('.'))
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded successfully (.h5)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {str(e)[:200]}")
        return None

            
    except Exception as e2:
        st.sidebar.error(f"❌ Manual loading failed: {str(e2)[:200]}")
            
            # METHOD 3: Last resort - load weights only
    try:
                st.sidebar.info("🔄 Trying weights-only loading...")
                
                # You'll need to know your model architecture
                # For EfficientNetB0:
                from tensorflow.keras.applications import EfficientNetB0
                from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
                from tensorflow.keras.models import Model
                
                # Recreate model architecture
                base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.5)(x)
                predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=predictions)
                
                # Load weights
                model.load_weights(model_path)
                
                st.sidebar.success("✅ Model loaded via weights-only")
                return model
                
    except Exception as e3:
                st.error(f"❌ All loading methods failed: {str(e3)[:200]}")
                return None

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess_image(image_obj):
    """Preprocess image for EfficientNet"""
    image = image_obj.convert("RGB")
    image = image.resize(IMG_SIZE)
    
    img_arr = np.array(image)
    img_arr = img_arr / 255.0  # Basic normalization
    
    # Try EfficientNet preprocessing
    try:
        img_arr = tf.keras.applications.efficientnet.preprocess_input(img_arr)
    except:
        # Fallback normalization
        img_arr = (img_arr - 0.5) * 2.0
    
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# -------------------------------
# CREATE OVERLAY IMAGE
# -------------------------------
def create_overlay(original_img, heatmap_array):
    """Overlay heatmap on original image"""
    original_resized = original_img.resize(IMG_SIZE)
    original_np = np.array(original_resized)
    
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_array, cv2.COLOR_BGR2RGB)
    
    # Blend images
    alpha = 0.5
    overlay = cv2.addWeighted(original_np, 1-alpha, heatmap_rgb, alpha, 0)
    
    return Image.fromarray(overlay)

# -------------------------------
# SIMPLE FALLBACK HEATMAP
# -------------------------------
def create_fallback_heatmap():
    """Create a simple heatmap when Grad-CAM fails"""
    size = 224
    heatmap = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create radial gradient
    center_y, center_x = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            intensity = max(0, 255 - dist)
            heatmap[y, x] = [intensity, 0, 0]  # Red gradient
    
    return heatmap

# -------------------------------
# MAIN APP
# -------------------------------
st.title("🔬 Skin Cancer Classifier")
st.write("Upload a skin lesion image for analysis")

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()

if model:
    st.success("✅ Model loaded successfully!")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    camera_img = st.camera_input("Or take a photo")
    
    source_image = None
    if uploaded_file:
        source_image = Image.open(uploaded_file)
    elif camera_img:
        source_image = Image.open(camera_img)
    
    if source_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(source_image, use_column_width=True)
        
        if st.button("🔬 Analyze with Grad-CAM", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                processed = preprocess_image(source_image)
                
                # Predict
                predictions = model.predict(processed, verbose=0)
                pred_idx = np.argmax(predictions[0])
                confidence = predictions[0][pred_idx]
                
                with col2:
                    st.subheader("📋 Diagnosis")
                    st.success(f"**{CLASS_NAMES[pred_idx].title()}**")
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Grad-CAM
                st.subheader("🔥 Grad-CAM Heatmap")
                
                try:
                    # Use YOUR Grad-CAM function
                    heatmap_array = generate_gradcam(model, processed)
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}. Using fallback heatmap.")
                    heatmap_array = create_fallback_heatmap()
                
                # Display heatmap and overlay
                col_heat, col_overlay = st.columns(2)
                
                with col_heat:
                    heatmap_rgb = cv2.cvtColor(heatmap_array, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_rgb, caption="AI Attention Heatmap", use_column_width=True)
                
                with col_overlay:
                    overlay_img = create_overlay(source_image, heatmap_array)
                    st.image(overlay_img, caption="Heatmap Overlay", use_column_width=True)
                
                # Probabilities
                st.subheader("📊 Probability Distribution")
                for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
                    st.progress(float(prob), text=f"{cls.title()}: {prob:.2%}")
else:
    st.error("""
    ## ❌ Model Loading Failed
    
    **Quick Fixes:**
    
    1. **Re-save your model with TensorFlow 2.15:**
    ```python
    # In your training code:
    model.save('skin_cancer_model.h5')  # Save as .h5 instead of .keras
    ```
    
    2. **Or upload via Streamlit:**
    """)
    
    # Allow model upload
    uploaded_model = st.file_uploader("Upload your model file", type=['h5', 'keras'])
    if uploaded_model:
        with open("skin_cancer_model.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("Model uploaded! Refresh the page.")
        st.rerun()

# Footer
st.warning("⚠️ For educational purposes only. Consult a doctor for diagnosis.")