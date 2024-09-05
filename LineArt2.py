import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET  # To handle SVG processing

# Function to apply sketch effect for pixel-based images
def apply_sketch_effect(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
    
    # Invert the blurred image
    inverted_blurred_image = cv2.bitwise_not(blurred_image)
    
    # Create the sketch effect by dividing the grayscale image by the inverted blurred image
    sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    
    return sketch_image

# Function to process pixel-based images
def process_pixel_image(uploaded_image):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Apply the sketch effect
    sketch_image = apply_sketch_effect(image)
    return sketch_image, image

# Function to process SVG by modifying stroke style to mimic a sketch effect
def process_svg_image(svg_data, new_stroke_width=2, dasharray="5,5"):
    # Parse the SVG XML
    tree = ET.ElementTree(ET.fromstring(svg_data))
    root = tree.getroot()
    
    # Define the namespace (SVG files have a namespace)
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Find all path elements and adjust stroke-width and stroke-dasharray for sketch effect
    for element in root.findall(".//svg:path", ns):
        if 'stroke' in element.attrib:
            element.set('stroke-width', str(new_stroke_width))
            element.set('stroke-dasharray', dasharray)

    # Convert the modified SVG back to string
    modified_svg_data = ET.tostring(root, encoding='unicode')
    return modified_svg_data

# Streamlit UI
st.title("Sketch Effect for SVG and Raster Images")
st.write("Upload your image (SVG, PNG, JPG) and apply a sketch effect.")

# Image Upload
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, SVG)", type=["png", "jpg", "jpeg", "svg"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ['png', 'jpg', 'jpeg']:
        # Handle pixel-based images
        sketch_image, original_image = process_pixel_image(uploaded_file)

        # Show original and sketch images
        if sketch_image is not None:
            st.image([original_image, sketch_image], caption=["Original Image", "Sketch Effect"], use_column_width=True)

        # Download the sketch effect as PNG
        buf = BytesIO()
        sketch_image_pil = Image.fromarray(sketch_image)
        sketch_image_pil.save(buf, format="PNG")

        st.download_button(
            label="Download Sketch Image as PNG",
            data=buf.getvalue(),
            file_name="sketch_image.png",
            mime="image/png"
        )

    elif file_type == 'svg':
        # Handle vector-based SVG images
        st.write("Processing as an SVG image.")
        
        # Read SVG content
        svg_data = uploaded_file.read().decode()

        # Slider for adjusting stroke width and dash array for sketch-like effect
        new_stroke_width = st.slider("Adjust stroke width", 1, 10, 2)
        dasharray = st.selectbox("Select dash pattern", options=["None", "5,5", "2,2", "10,5"])

        if dasharray == "None":
            dasharray = ""

        # Process the SVG by modifying stroke properties
        modified_svg_data = process_svg_image(svg_data, new_stroke_width, dasharray)

        # Display the modified SVG as text (optionally render it)
        st.text_area("Modified SVG Data", modified_svg_data, height=300)

        # Download the modified SVG
        st.download_button(
            label="Download Modified SVG",
            data=modified_svg_data,
            file_name="sketch_svg_image.svg",
            mime="image/svg+xml"
        )
