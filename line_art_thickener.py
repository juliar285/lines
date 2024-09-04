import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Function to process the image
def process_image(uploaded_image, thickness=0.1, blur_intensity=0):
    # Convert the uploaded image to an OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Apply dilation to thicken the edges based on the slider value
    kernel_size = int(thickness * 10)  # Directly scale the kernel size for visible changes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply anti-aliasing (optional Gaussian blur)
    if blur_intensity > 0:
        smoothed_edges = cv2.GaussianBlur(thickened_edges, (blur_intensity, blur_intensity), 0)
    else:
        smoothed_edges = thickened_edges

    # Create a copy of the original image and apply the thickened black edges
    image_with_black_edges = image.copy()
    image_with_black_edges[smoothed_edges != 0] = [0, 0, 0]  # Set thickened edges to black

    return image_with_black_edges, image  # Return the processed and original images

# Streamlit UI
st.title("Line Art Thickener with Adjustable Blur")
st.write("Upload your line art, adjust the line thickness and blur settings, and we'll process the image for you!")

# Upload the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Slider to control line thickness
    thickness = st.slider("Select line thickness", 0.01, 5.0, 0.1, step=0.01)
    
    # Slider to control blur intensity (0 means no blur)
    blur_intensity = st.slider("Select blur intensity (0 for no blur)", 0, 10, 0)

    # Process the image
    processed_image, original_image = process_image(uploaded_image, thickness, blur_intensity)
    
    # Show both images side by side for comparison
    st.image([original_image, processed_image], caption=["Original Image", "Processed Image"], use_column_width=True)

    # Option to accept the processed image
    if st.button('Accept Processed Image'):
        st.success("You have accepted the processed image!")
        
        # Provide download option
        buf = BytesIO()
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        processed_image_pil.save(buf, format="PNG")
        st.download_button(label="Download Processed Image", data=buf.getvalue(), file_name="processed_image.png", mime="image/png")
    else:
        st.warning("You haven't accepted the processed image yet.")
