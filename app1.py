import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deep Image Droid", page_icon="🛡️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {background-color: #0e1117; color: white; font-family: 'Segoe UI', sans-serif;}
h1, h2, h3 {color:#00ffcc;}
button {background-color:#00cc66; color:white; border-radius:8px; padding:8px 20px;}
.result-box {padding: 20px; border-radius: 10px; text-align: center; font-size: 20px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<h1 style="text-align:center">🛡️ Deep Image Droid Malware Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:18px; color:#cccccc;">Upload → Analyze → Detect Threats in Seconds</p>', unsafe_allow_html=True)
st.write("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Upload & Detect", "About Project"])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ---------------- HOME ----------------
if menu == "Home":
    st.subheader("🚀 Welcome")
    st.write("This system detects whether an image is Malware or Benign using Deep Learning.")
    st.info("💡 Try uploading a sample image from the next tab!")

# ---------------- UPLOAD & SAMPLE ----------------
elif menu == "Upload & Detect":
    st.subheader("📂 Upload & Detect")

    tab1, tab2 = st.tabs(["Upload Image", "Try Sample Images"])

    with tab1:
        file = st.file_uploader("Upload APK Visualization Image", type=["jpg","png","jpeg"])

st.info("Upload only APK-derived binary visualization images.")

file = st.file_uploader("Upload APK Visualization Image", type=["jpg", "jpeg", "png"])

st.info("Upload only APK-derived or binary-visualization style images.")

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    img_input = np.expand_dims(img, axis=0)

    prediction = model.predict(img_input)

    if prediction[0][0] > 0.5:
        st.error(f"Malware-like Pattern Detected ({prediction[0][0]*100:.2f}%)")
    else:
        st.success(f"Benign-like Pattern Detected ({(1-prediction[0][0])*100:.2f}%)")
            
        
        
    with tab2:
        st.write("### Try Sample Images")
        col1, col2 = st.columns(2)
        if col1.button("Sample Malware Image"):
            sample_img = Image.open("sample_images/malware1.jpg")
            st.image(sample_img, caption="Sample Malware", use_column_width=True)
            # Run detection same as above
        if col2.button("Sample Benign Image"):
            sample_img = Image.open("sample_images/benign1.jpg")
            st.image(sample_img, caption="Sample Benign", use_column_width=True)
            # Run detection same as above

# ---------------- ABOUT ----------------
elif menu == "About Project":
    st.subheader("🧠 How It Works")
    st.write("""
    1. Image is uploaded by user
    2. Image is resized and normalized
    3. CNN model extracts features
    4. Model classifies as Malware or Benign
    """)
    st.write("---")
    st.subheader("📊 Model Details")
    st.write("- Model: Convolutional Neural Network (CNN)")
    st.write("- Input Size: 128x128")
    st.write("- Output: Binary Classification")
    st.success("✔️ Accuracy: ~95% (example)")
