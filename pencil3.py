import streamlit as st
from PIL import Image
import vtracer
import io
import numpy as np

# Streamlit app layout
st.title("Image to SVG Converter with Adjustable Parameters")

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

    # Step 3: Use vtracer to convert the image bytes to SVG
    # Convert the image bytes into SVG string using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes, 
        img_format='png',  # Format is PNG as the input image is treated as PNG
        colormode=colormode, 
        path_precision=path_precision, 
        color_precision=color_precision, 
        layer_difference=layer_difference
    )

    # Step 4: Display the SVG output
    st.write("### Converted SVG:")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Step 5: Provide a download button for the SVG result
    st.download_button(label="Download SVG", data=svg_str, file_name="converted_image.svg", mime="image/svg+xml")
