import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("SVG Pencil Sketch Conversion with Adjustable Noise Removal")

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
    inverted_image = 255 - gray_image
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blur = 255 - blurred
    pencil_sketch_image = cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Step 4: Set a fixed threshold for line detection
    threshold_value = 244

    # Step 5: Convert to binary (black and white) image with fixed threshold
    _, binary_image = cv2.threshold(pencil_sketch_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Step 6: Allow the user to adjust the contour area threshold
    min_contour_area = st.slider("Minimum Contour Area for Noise Removal", 1, 200, 10)

    # Step 7: Apply noise removal using Morphological Operations (smaller kernel)
    kernel = np.ones((.1, .1), np.uint8)  # A smaller kernel for softer noise removal
    clean_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Step 8: Remove small contours based on the user-defined contour area threshold
    contours, _ = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_clean_image = np.zeros_like(clean_image)  # Create an empty image to store the cleaned result
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # Filter based on user-defined area
            cv2.drawContours(final_clean_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Convert the cleaned image back to PNG
    sketch_pil_image = Image.fromarray(final_clean_image)
    png_buffer = io.BytesIO()
    sketch_pil_image.save(png_buffer, format='PNG')
    png_data = png_buffer.getvalue()

    # Step 9: Display the cleaned black-and-white pencil sketch (PNG)
    st.write("### Cleaned Pencil Sketch (Black and White PNG):")
    st.image(final_clean_image, channels="GRAY", use_column_width=True)

    # Add a download button for the cleaned PNG version of the pencil sketch
    st.download_button(label="Download Cleaned Pencil Sketch (Black & White PNG)", data=png_data, file_name="cleaned_pencil_sketch_black_white.png", mime="image/png")

    # Step 10: Convert the cleaned black-and-white PNG to SVG using vtracer
    svg_sketch_str = vtracer.convert_raw_image_to_svg(png_data, img_format='png')

    # Display the SVG output of the black-and-white pencil sketch using HTML embedding
    st.write("### Cleaned Pencil Sketch SVG:")
    st.write(f'<div>{svg_sketch_str}</div>', unsafe_allow_html=True)

    # Provide a download option for the black-and-white pencil sketch SVG
    st.download_button(label="Download Cleaned Pencil Sketch (Black & White SVG)", data=svg_sketch_str, file_name="cleaned_pencil_sketch_black_white.svg", mime="image/svg+xml")
