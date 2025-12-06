import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------
# Streamlit Page Config
# -------------------------------------
st.set_page_config(
    page_title="OsteoVision - AI Bone Health Analyzer",
    layout="centered",
    page_icon="🧬"
)

IMAGE_SIZE = 224   # use same size as model was trained on
CLASS_NAMES = ['normal', 'osteoporosis']

# -------------------------------------
# Load Trained Model Safely
# -------------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("osteoporosis_model.h5", compile=False)
    return model

model = load_trained_model()

# -------------------------------------
# UI Styling
# -------------------------------------
st.markdown(
    """
    <style>
    .title { font-size: 2.5em; font-weight: bold; color: #2E86C1; text-align: center; }
    .subtitle { font-size: 1.2em; color: #555; text-align: center; margin-bottom: 30px; }
    .result-card {
        padding: 20px; background-color: #eaf2f8;
        border-radius: 12px; border-left: 5px solid #2E86C1;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🧬 OsteoVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Radiograph Analyzer for Osteoporosis Detection</div>', unsafe_allow_html=True)

# -------------------------------------
# Preprocess Uploaded Image
# -------------------------------------
def preprocess_image(uploaded_file):
    img = load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode="rgb")
    img_array = img_to_array(img)

    # Normalize if model was trained with normalization
    img_array = img_array / 255.0 

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------
# FILE UPLOADER
# -------------------------------------
uploaded_file = st.file_uploader("📤 Upload your radiographic image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess for model
    img = preprocess_image(uploaded_file)

    # Predict
    prediction = model.predict(img)[0][0]  # sigmoid output
    
    osteo_prob = float(prediction)
    normal_prob = 1 - osteo_prob
    
    if osteo_prob > 0.5:
        final_label = "Osteoporosis"
        confidence = osteo_prob
    else:
        final_label = "Normal"
        confidence = normal_prob

    # -------------------------------------
    # RESULT CARD
    # -------------------------------------
    st.markdown(f"""
    <div class="result-card">
        <h4>🧠 Prediction Results</h4>
        <p><strong>Condition:</strong> {final_label}</p>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------
    # Probability Bar Chart
    # -------------------------------------
    fig, ax = plt.subplots()
    ax.bar(["Normal", "Osteoporosis"], [normal_prob, osteo_prob])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Distribution")
    st.pyplot(fig)

st.markdown("---")
st.caption("⚠️ This tool assists analysis but does not replace professional medical evaluation.")
