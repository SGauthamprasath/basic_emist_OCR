import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Function to preprocess image
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to array and normalize
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img_array

# Streamlit app
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to predict its value.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=280)
    
    # Predict from uploaded image
    if st.button("Predict"):
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit])
        
        st.write(f"*Predicted Digit*: {predicted_digit} (Confidence: {confidence * 100:.2f}%)")
    else:
        st.write("Click 'Predict' to get the result.")
else:
    st.write("Please upload an image to proceed.")