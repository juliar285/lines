import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("Direct SVG Conversion with Adjustable Threshold and Noise Reduction")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Step 1: Convert the uploaded image to bytes
    image = Image.open(uploaded_file)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Step 2: Convert image bytes to OpenCV format for processing
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)

    # Step 3: Apply pencil sketch effect using OpenCV
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to invert and soften
    inverted_image = 255 - gray_image
    blurred = cv2.GaussianBlur(inverted_image, (15, 15), 0)
    inverted_blur = 255 - blurred
    pencil_sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Step 4: Apply a Bilateral Filter to reduce noise while preserving details
    bilateral_filtered = cv2.bilateralFilter(pencil_sketch_image, d=5, sigmaColor=50, sigmaSpace=50)

    # Step 5: Add a threshold slider for fine-tuning the binary conversion
    threshold_value = st.slider("Threshold Value for Binary Conversion", 50, 255, 244)

    # Apply the final threshold to convert to black-and-white (binary)
    _, binary_image = cv2.threshold(bilateral_filtered, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 6: Convert the final binary image to bytes for SVG conversion
    sketch_pil_image = Image.fromarray(binary_image)
    img_byte_array = io.BytesIO()
    sketch_pil_image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Step 7: Convert the processed image to SVG using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(img_bytes, img_format='png')

    # Step 8: Display the final SVG output with adjustable threshold and noise reduction
    st.write("### Pencil Sketch SVG with Adjustable Threshold and Noise Reduction:")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Provide a download option for the SVG file
    st.download_button(label="Download Pencil Sketch (SVG)", data=svg_str, file_name="pencil_sketch.svg", mime="image/svg+xml")
