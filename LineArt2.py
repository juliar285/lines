import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import xml.etree.ElementTree as ET  # To handle SVG processing
import base64

# Function to convert SVG to PNG for preview
def svg_to_png(svg_data):
    # Use Pillow to create a blank canvas and render SVG
    # Note: Streamlit Cloud does not support libraries like cairosvg, so you may need a workaround for SVG to PNG conversion
    return None  # Placeholder for converting SVG to PNG

# Function to process SVG by modifying stroke width
def process_svg_image(svg_data, new_stroke_width=2):
    # Parse the SVG XML
    tree = ET.ElementTree(ET.fromstring(svg_data))
    root = tree.getroot()
    
    # Define the namespace (SVG files have a namespace)
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Find all path elements and adjust stroke-width
    for element in root.findall(".//svg:path", ns):
        if 'stroke' in element.attrib:
            element.set('stroke-width', str(new_stroke_width))

    # Convert the modified SVG back to string
    modified_svg_data = ET.tostring(root, encoding='unicode')
    return modified_svg_data

# Streamlit UI
st.title("SVG Stroke Width Adjuster with Preview")

# Image Upload
uploaded_file = st.file_uploader("Upload an image (SVG)", type=["svg"])

if uploaded_file is not None:
    # Read SVG content
    svg_data = uploaded_file.read().decode()

    # Slider for adjusting stroke width
    new_stroke_width = st.slider("Adjust stroke width", 1, 10, 2)

    # Process the SVG by modifying stroke properties
    modified_svg_data = process_svg_image(svg_data, new_stroke_width)

    # Try converting the modified SVG to PNG for display
    st.write("### Preview of Modified SVG as Image (PNG):")
    png_image = svg_to_png(modified_svg_data)

    if png_image:
        st.image(png_image, caption="Modified SVG as PNG")

    # Display the modified SVG as text (optional, for debugging purposes)
    st.text_area("Modified SVG Data", modified_svg_data, height=300)

    # Download the modified SVG
    st.download_button(
        label="Download Modified SVG",
        data=modified_svg_data,
        file_name="modified_svg_image.svg",
        mime="image/svg+xml"
    )
