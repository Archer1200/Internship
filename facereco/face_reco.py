import pickle
import streamlit as st
import numpy as np
from PIL import Image


model = pickle.load(open('face_reco.pkl', 'rb'))

def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((64, 64))  
    img_array = np.array(img)
    flat_img = img_array.flatten().reshape(1, -1)  
    return flat_img

def deploy():
    st.title("Face Recognition System")

    test = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])    
    pred = st.button('Predict')

    if test is not None:
        st.image(test, caption="Uploaded Image", use_container_width=True)
        processed= preprocess_image(test)
        x=model.predict(processed)
        st.write(f"Prediction: {x[0]}")
    
deploy()
