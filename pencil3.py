import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2
import cairosvg  # For handling SVG images

# Streamlit app layout
st.title("Image to Color SVG Converter with Adjustable Pencil Sketch Effect")

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

    # Step 2: Adjust vtracer conversion parameters using sliders
    st.sidebar.title("SVG Conversion Parameters")

    # Sliders for vtracer parameters
    path_precision = st.sidebar.slider("Path Precision", 1, 10, 6)
    color_precision = st.sidebar.slider("Color Precision", 1, 10, 6)
    layer_difference = st.sidebar.slider("Layer Difference", 1, 50, 16)

    # Convert the image to a color SVG using vtracer with adjustable parameters
    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes, 
        img_format='png',  # Treat the image as PNG
        colormode="color",  # Always convert to color
        path_precision=path_precision,  # Adjustable precision
        color_precision=color_precision,  # Adjustable color precision
        layer_difference=layer_difference  # Adjustable layer difference
    )

    # Step 3: Display the converted color SVG
    st.write("### Converted Color SVG:")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Step 4: Download the color SVG
    st.download_button(label="Download Color SVG", data=svg_str, file_name="color_image.svg", mime="image/svg+xml")

    # Step 5: Ask if the user wants to apply the pencil sketch effect
    apply_sketch = st.checkbox("Apply Pencil Sketch Effect to SVG", value=False)

    if apply_sketch:
        # Step 6: Adjust pencil sketch parameters using sliders
        st.sidebar.title("Pencil Sketch Parameters")

        # Adjust Gaussian blur level
        blur_amount = st.sidebar.slider("Blur Amount", 1, 50, 21, step=2)

        # Adjust threshold for sketch effect
        threshold_value = st.sidebar.slider("Sketch Threshold", 0, 255, 128)

        # Adjust noise reduction strength (bilateral filter)
        noise_reduction = st.sidebar.slider("Noise Reduction Strength", 1, 100, 50)

        # Step 7: Convert the SVG to PNG for processing
        png_image = cairosvg.svg2png(bytestring=svg_str)

        # Convert PNG to OpenCV format for processing
        nparr = np.frombuffer(png_image, np.uint8)
        open_cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Apply pencil sketch effect using OpenCV
        inverted_image = 255 - gray_image
        blurred = cv2.GaussianBlur(inverted_image, (blur_amount, blur_amount), 0)
        inverted_blur = 255 - blurred
        pencil_sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)

        # Optionally apply noise reduction using bilateral filtering
        pencil_sketch_image = cv2.bilateralFilter(pencil_sketch_image, d=9, sigmaColor=noise_reduction, sigmaSpace=noise_reduction)

        # Apply threshold to convert to binary (for sketch)
        _, binary_sketch = cv2.threshold(pencil_sketch_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Display the pencil sketch effect
        st.image(binary_sketch, caption='Pencil Sketch Effect', use_column_width=True)

        # Convert the sketch image to PNG for download
        sketch_pil_image = Image.fromarray(binary_sketch)
        img_byte_array = io.BytesIO()
        sketch_pil_image.save(img_byte_array, format='PNG')
        sketch_bytes = img_byte_array.getvalue()

        # Step 8: Provide a download button for the pencil sketch PNG result
        st.download_button(label="Download Pencil Sketch PNG", data=sketch_bytes, file_name="pencil_sketch.png", mime="image/png")
