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

# --- Webcam Logic ---
if run_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera permissions.")
    else:
        st.info("Place your hand inside the green box. Press the checkbox again to stop.")
        frame_placeholder = st.empty()

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            frame = cv2.flip(frame, 1)

            # Define the Region of Interest (ROI)
            box_size = 300
            height, width, _ = frame.shape
            x1 = int((width - box_size) / 2)
            y1 = int((height - box_size) / 2)
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Draw the ROI box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract and preprocess the ROI
            roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            resized_roi = cv2.resize(rgb_roi, IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(resized_roi)
            img_array = tf.expand_dims(img_array, 0)

            # Make prediction
            with st.spinner('Predicting...'):
                predictions = model.predict(img_array, verbose=0)
            
            predicted_index = np.argmax(predictions[0])
            predicted_class = class_names[predicted_index]
            confidence = 100 * np.max(predictions[0])

            # Display prediction on the frame
            display_text = f"Prediction: {predicted_class} ({confidence:.1f}%)"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame in Streamlit
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

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
