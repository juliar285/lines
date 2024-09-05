import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np
import cv2

# Streamlit app layout
st.title("SVG Pencil Sketch Conversion")

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

    # Convert the pencil sketch image back to PNG
    sketch_pil_image = Image.fromarray(pencil_sketch_image)
    img_byte_array = io.BytesIO()
    sketch_pil_image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Step 4: Display the PNG version of the pencil sketch
    st.write("### Pencil Sketch (PNG):")
    st.image(pencil_sketch_image, channels="GRAY", use_column_width=True)

    # Add a download button for the PNG version of the pencil sketch
    st.download_button(label="Download Pencil Sketch (PNG)", data=img_bytes, file_name="pencil_sketch.png", mime="image/png")

    # Step 5: Convert the pencil sketch (PNG) to SVG using vtracer
    svg_sketch_str = vtracer.convert_raw_image_to_svg(img_bytes, img_format='png')

    # Display the SVG output of the pencil sketch using HTML embedding
    st.write("### Pencil Sketch SVG:")
    st.write(f'<div>{svg_sketch_str}</div>', unsafe_allow_html=True)

    # Provide a download option for the pencil sketch SVG
    st.download_button(label="Download Pencil Sketch (SVG)", data=svg_sketch_str, file_name="pencil_sketch.svg", mime="image/svg+xml")
