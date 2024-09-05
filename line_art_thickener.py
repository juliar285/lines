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
st.title("Line Art Thickener with 300 DPI Output")
st.write("Upload your line art, adjust the line thickness, and ensure the final image is saved at 300 DPI!")

# Create an empty placeholder for the file uploader
uploader_placeholder = st.empty()

# Upload the image using the placeholder
uploaded_image = uploader_placeholder.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Slider to control line thickness
    thickness = st.slider("Select line thickness", 0.5, 1.0, 0.5, step=0.01)
    
    # Slider to control the upscaling factor for smoother processing
    upscale_factor = st.slider("Upscale factor (higher values reduce pixelation)", 1, 6, 2)

    # Process the image
    processed_image, original_image = process_image(uploaded_image, thickness, upscale_factor)
    
    # Show both images side by side for comparison
    st.image([original_image, processed_image], caption=["Original Image", "Processed Image"], use_column_width=True)

    # Option to accept the processed image
    if st.button('Accept Processed Image'):
        st.success("You have accepted the processed image!")
        
        # Provide download option with 300 DPI
        buf = BytesIO()
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        processed_image_pil.save(buf, format="PNG", dpi=(300, 300))  # Save at 300 DPI
        
        # Add download button
        st.download_button(label="Download Processed Image at 300 DPI", data=buf.getvalue(), file_name="processed_image_300dpi.png", mime="image/png")

        # After the download, clear the file uploader to reset the UI
        uploader_placeholder.empty()  # Clears the uploader and resets the app
else:
    st.warning("Please upload an image to proceed.")
