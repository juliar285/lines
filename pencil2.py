import streamlit as st
from PIL import Image
import vtracer
import io

# Streamlit app layout
st.title("Image to SVG Conversion with vtracer")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the uploaded image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Convert the image bytes to SVG using vtracer
    svg_str = vtracer.convert_raw_image_to_svg(img_bytes, img_format='png')

    # Display the SVG output using HTML embedding
    st.write("### Converted SVG:")
    st.write(f'<div>{svg_str}</div>', unsafe_allow_html=True)

    # Download option for the generated SVG
    st.download_button(label="Download SVG", data=svg_str, file_name="output.svg", mime="image/svg+xml")
