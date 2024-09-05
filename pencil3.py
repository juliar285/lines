import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("Conservative SVG Pencil Sketch Conversion")

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

    # Step 2: Convert the image bytes to SVG using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(img_bytes, img_format='png')

    # Step 3: Convert SVG back to raster image (PNG) using OpenCV for sketching
    png_image = Image.open(io.BytesIO(img_bytes))
    open_cv_image = np.array(png_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)

    # Apply pencil sketch effect using OpenCV
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Adjust Gaussian blur for a more conservative effect
    inverted_image = 255 - gray_image
    blurred = cv2.GaussianBlur(inverted_image, (15, 15), 0)  # Reduced blur kernel size
    inverted_blur = 255 - blurred
    pencil_sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Step 4: Allow the user to adjust the threshold value
    threshold_value = st.slider("Threshold Value for Line Detection", 50, 255, 200)  # Lower default threshold

    # Step 5: Convert to binary (black and white) image with adjustable threshold
    _, binary_image = cv2.threshold(pencil_sketch_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Skipping morphological operations for now to preserve more details

    # Convert the binary image back to PNG
    sketch_pil_image = Image.fromarray(binary_image)
    png_buffer = io.BytesIO()
    sketch_pil_image.save(png_buffer, format='PNG')
    png_data = png_buffer.getvalue()

    # Step 6: Display the cleaned black-and-white pencil sketch (PNG)
    st.write("### Pencil Sketch (Black and White PNG):")
    st.image(binary_image, channels="GRAY", use_column_width=True)

    # Add a download button for the cleaned PNG version of the pencil sketch
    st.download_button(label="Download Pencil Sketch (Black & White PNG)", data=png_data, file_name="pencil_sketch_black_white.png", mime="image/png")

    # Step 7: Convert the black-and-white PNG to SVG using vtracer
    svg_sketch_str = vtracer.convert_raw_image_to_svg(png_data, img_format='png')

    # Display the SVG output of the black-and-white pencil sketch using HTML embedding
    st.write("### Pencil Sketch SVG:")
    st.write(f'<div>{svg_sketch_str}</div>', unsafe_allow_html=True)

    # Provide a download option for the black-and-white pencil sketch SVG
    st.download_button(label="Download Pencil Sketch (Black & White SVG)", data=svg_sketch_str, file_name="pencil_sketch_black_white.svg", mime="image/svg+xml")
