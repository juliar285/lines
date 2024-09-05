import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("Image to SVG Converter with Pencil Sketch Effect")

# Step 1: Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "svg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to bytes for processing
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Step 2: Adjust conversion parameters using sliders
    st.sidebar.title("SVG Conversion Parameters")
    
    # Select color mode
    colormode = st.sidebar.selectbox("Color Mode", ["color", "binary"])
    
    # Adjust path precision
    path_precision = st.sidebar.slider("Path Precision", 1, 10, 6)

    # Thresholds for details
    color_precision = st.sidebar.slider("Color Precision", 1, 10, 6)
    layer_difference = st.sidebar.slider("Layer Difference", 1, 50, 16)

    # Option to apply pencil sketch effect or not
    apply_sketch = st.sidebar.checkbox("Apply Pencil Sketch Effect", value=True)

    # Step 3: Process the image in OpenCV if applying the pencil sketch effect
    if apply_sketch:
        # Convert the image to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Apply pencil sketch effect using OpenCV
        inverted_image = 255 - gray_image
        blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
        inverted_blur = 255 - blurred
        pencil_sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)

        # Convert the sketch image to bytes for SVG conversion
        sketch_pil_image = Image.fromarray(pencil_sketch_image)
        img_byte_array = io.BytesIO()
        sketch_pil_image.save(img_byte_array, format='PNG')
        img_bytes = img_byte_array.getvalue()

    # Step 4: Convert the image (or sketch) to SVG using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes, 
        img_format='png',  # Treat the image as PNG
        colormode=colormode, 
        path_precision=path_precision, 
        color_precision=color_precision, 
        layer_difference=layer_difference
    )

    # Step 5: Display the SVG output
    st.write("### Converted SVG:")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Step 6: Provide a download button for the SVG result
    st.download_button(label="Download SVG", data=svg_str, file_name="converted_image.svg", mime="image/svg+xml")
