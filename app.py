import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(page_title="ASL Hand Sign Detection", layout="wide")

# Function to load the model and class names
@st.cache_resource
def load_model_and_classes():
    """Loads the trained model and class names."""
    try:
        model = tf.keras.models.load_model('asl_model.h5')
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names: {e}")
        st.error("Please make sure 'asl_model.h5' and 'class_names.txt' are in the same directory.")
        st.stop()

model, class_names = load_model_and_classes()

# Constants
IMG_SIZE = (64, 64)

# --- UI Layout ---
st.title("ASL Hand Sign Detector")
st.markdown("This application uses a Convolutional Neural Network to detect American Sign Language hand signs in real-time or from an uploaded image.")

col1, col2 = st.columns(2)

with col1:
    st.header("Live Webcam Detection")
    run_webcam = st.checkbox('Start Webcam')

with col2:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Webcam Logic --- (ADD THIS NEW BLOCK)
with col1: # This keeps it in the left column
    st.info("Click the button below to capture an image from your webcam.")
    img_file_buffer = st.camera_input("Capture Image for Prediction")

    if img_file_buffer is not None:
        # To read image file buffer as bytes:
        bytes_data = img_file_buffer.getvalue()
        # Convert the bytes data to a numpy array
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Display the captured image
        st.image(cv2_img, channels="BGR", caption='Your Captured Image')
        st.markdown("---")

        with st.spinner('Analyzing image...'):
            # Preprocess the image for the model
            rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_img, IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(resized_img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = 100 * np.max(predictions[0])

            st.success(f"**Predicted Sign:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
# --- Image Upload Logic ---
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert bytes data to a numpy array
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with col2:
        st.image(cv2_img, channels="BGR", caption='Uploaded Image.', use_column_width=True)
        st.markdown("---")

        if st.button('Predict Sign from Image'):
            with st.spinner('Analyzing image...'):
                # Preprocess the image for the model
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(rgb_img, IMG_SIZE)
                img_array = tf.keras.utils.img_to_array(resized_img)
                img_array = tf.expand_dims(img_array, 0)
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = 100 * np.max(predictions[0])

                st.success(f"**Predicted Sign:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")
