import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Set page configuration
st.set_page_config(page_title="ASL Hand Sign Detection", layout="wide")

# Function to load the model and class names (no changes needed here)
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

# This is the core of the real-time processing
class ASLVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.class_names = class_names

    def recv(self, frame):
        # Convert the frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # Define the Region of Interest (ROI)
        box_size = 300
        height, width, _ = img.shape
        x1 = int((width - box_size) / 2)
        y1 = int((height - box_size) / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Draw the ROI box on the original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract and preprocess the ROI
        roi = img[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized_roi = cv2.resize(rgb_roi, IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(resized_roi)
        img_array = tf.expand_dims(img_array, 0)

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_index]
        confidence = 100 * np.max(predictions[0])

        # Display prediction on the frame
        display_text = f"Prediction: {predicted_class} ({confidence:.1f}%)"
        cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# --- UI Layout ---
st.title("ASL Hand Sign Detector")
st.markdown("This application uses a Convolutional Neural Network to detect American Sign Language hand signs in real-time or from an uploaded image.")

col1, col2 = st.columns(2)

with col1:
    st.header("Live Webcam Detection")
    st.info("Click 'START' to begin the real-time detection. Place your hand inside the green box.")
    
    # RTCConfiguration is needed for deployment on Streamlit Cloud
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="asl-detection",
        video_transformer_factory=ASLVideoTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # --- Image Upload Logic (no changes needed here) ---
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        st.image(cv2_img, channels="BGR", caption='Uploaded Image.', use_column_width=True)
        st.markdown("---")

        if st.button('Predict Sign from Image'):
            with st.spinner('Analyzing image...'):
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(rgb_img, IMG_SIZE)
                img_array = tf.keras.utils.img_to_array(resized_img)
                img_array = tf.expand_dims(img_array, 0)
                
                predictions = model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_index]
                confidence = 100 * np.max(predictions[0])

                st.success(f"**Predicted Sign:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")
