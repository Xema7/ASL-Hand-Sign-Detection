import streamlit as st

st.title("Native Streamlit Camera Test")
st.info("This test uses st.camera_input, which is the most reliable way to access a camera in a deployed Streamlit app.")

# This widget is designed to work directly with browser permissions.
img_file_buffer = st.camera_input("Click here to take a picture")

if img_file_buffer is not None:
    st.success("Success! The camera is working.")
    st.image(img_file_buffer)
