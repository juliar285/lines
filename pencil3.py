import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("SVG Conversion with Edge Detection and Contour Drawing")

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

    # Step 2: Convert the image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)

    # Step 3: Convert the image to grayscale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Canny Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Step 5: Find contours based on the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Draw the contours on a blank image (same size as original)
    contour_image = np.zeros_like(gray_image)  # Create a blank image
    cv2.drawContours(contour_image, contours, -1, (255), thickness=2)  # Draw contours in white

    # Step 7: Apply a threshold slider to fine-tune the binary conversion
    threshold_value = st.slider("Threshold Value for Edge Detection", 50, 255, 200)
    _, binary_image = cv2.threshold(contour_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 8: Convert the contour-drawn image back to PNG format
    sketch_pil_image = Image.fromarray(binary_image)
    img_byte_array = io.BytesIO()
    sketch_pil_image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Step 9: Convert the final image with contours to SVG using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(img_bytes, img_format='png')

    # Step 10: Display the SVG output
    st.write("### Contour-drawn SVG (with Adjustable Edge Detection):")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Provide a download option for the SVG file
    st.download_button(label="Download SVG", data=svg_str, file_name="contour_sketch.svg", mime="image/svg+xml")
