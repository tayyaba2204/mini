import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("Deepimagedroid Malware Detector")

model = tf.keras.models.load_model("model.h5")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")  
    st.image(image)

    img = np.array(image)
    img = cv2.resize(img,(128,128)) / 255.0
    img = img.reshape(1,128,128,3)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("Malware Detected")
    else:
        st.success("Safe App")