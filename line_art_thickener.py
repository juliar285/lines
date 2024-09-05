import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Function to process the image
def process_image(uploaded_image, thickness=0.5, upscale_factor=2):
    # Convert the uploaded image to an OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # If the image has an alpha channel (e.g., PNG), convert it to RGB
    if image.shape[2] == 4:  # Check for RGBA (PNG with transparency)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Upscale the image for smoother processing
    image_upscaled = cv2.resize(image, (0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_upscaled, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Apply dilation to thicken the edges based on the slider value
    kernel_size = int(thickness * 10)  # Directly scale the kernel size for visible changes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply bilateral filtering to reduce pixelation while keeping edges sharp
    smoothed_edges = cv2.bilateralFilter(thickened_edges, d=9, sigmaColor=75, sigmaSpace=75)

    # Create a copy of the upscaled image and apply the thickened black edges
    image_with_black_edges = image_upscaled.copy()
    image_with_black_edges[smoothed_edges != 0] = [0, 0, 0]  # Set thickened edges to black

    # Downscale the image back to the original resolution
    result_image = cv2.resize(image_with_black_edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    return result_image, image  # Return the processed and original images

# Streamlit UI
st.title("Bold and Consistent Line Art")
st.write("Upload your line art, adjust the line thickness, and get a processed image!")

# Initialize session state for tracking the uploaded file and processed image
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

# Reset the session state for the uploader and image
def reset_session():
    st.session_state.uploaded_image = None
    st.session_state.processed_image = None

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="file_uploader")

if uploaded_image is not None:
    st.session_state.uploaded_image = uploaded_image  # Store the uploaded image in session state

# Only proceed if an image has been uploaded
if st.session_state.uploaded_image:
    # Get the file extension (for download format matching)
    file_extension = os.path.splitext(uploaded_image.name)[1].lower()  # ".png" or ".jpg"
    download_format = "PNG" if file_extension == ".png" else "JPEG"

    # Slider to control line thickness with a default of 0.5 and a range from 0.01 to 1.0
    thickness = st.slider("Select line thickness", 0.01, 1.0, 0.5, step=0.01)
    
