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

# --- Reusable Prediction Function ---
def predict_sign(image_np):
    """
    Takes a numpy array image (in BGR format from OpenCV), preprocesses it,
    and returns the predicted class and confidence score.
    """
    # Preprocess the image for the model
    rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(resized_img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make prediction
    with st.spinner('Predicting...'):
        predictions = model.predict(img_array, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])
    
    return predicted_class, confidence

# --- UI Layout ---
st.title("ASL Hand Sign Detector")
st.markdown("This application uses a Convolutional Neural Network to detect American Sign Language hand signs from a captured photo or an uploaded image.")

col1, col2 = st.columns(2)

# --- Webcam Capture Logic ---
with col1:
    st.header("Capture from Webcam")
    # Use st.camera_input to provide a "Take photo" button
    camera_image = st.camera_input(
        "Position your hand and take a photo", 
        key="camera_capture"
    )

    # When a photo is taken, camera_image is no longer None
    if camera_image is not None:
        # Read the image bytes and convert to a numpy array for OpenCV
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        st.markdown("---")
        st.write("Analyzing captured image...")
        
        # Get and display the prediction
        predicted_class, confidence = predict_sign(cv2_img)
        st.success(f"**Predicted Sign:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

# --- Image Upload Logic ---
with col2:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file and convert to a numpy array
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(cv2_img, channels="BGR", caption='Uploaded Image.', use_column_width=True)
        st.markdown("---")

        # Add a button to trigger prediction
        if st.button('Predict Sign from Image'):
            # Get and display the prediction
            predicted_class, confidence = predict_sign(cv2_img)
            st.success(f"**Predicted Sign:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")