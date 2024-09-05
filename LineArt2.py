import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET  # To handle SVG processing

# Function to convert SVG to pixel-based (PNG)
def convert_svg_to_png(svg_data):
    # Handle SVG conversion, you may need a library or external service here
    # Since we cannot use cairosvg in Streamlit Cloud, provide a message
    st.error("SVG support is limited on this platform. Please upload PNG/JPG.")
    return None

# Function to process pixel-based images
def process_pixel_image(uploaded_image, thickness=0.5, upscale_factor=2):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Upscale the image for smoother processing
    image_upscaled = cv2.resize(image, (0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_upscaled, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Apply dilation to thicken the edges based on the slider value
    kernel_size = int(thickness * 10)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply bilateral filtering to reduce pixelation while keeping edges sharp
    smoothed_edges = cv2.bilateralFilter(thickened_edges, d=9, sigmaColor=75, sigmaSpace=75)

    return smoothed_edges, image

# Function to process SVG files by modifying stroke-width
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
st.title("Image Processor for Raster and Vector Formats")
st.write("Upload your image (SVG, PNG, JPG) and adjust parameters for processing.")

# Image Upload
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, SVG)", type=["png", "jpg", "jpeg", "svg"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ['png', 'jpg', 'jpeg']:
        # Handle pixel-based images
        st.write("Processing as a raster image (PNG/JPG).")
        # Slider for line thickness and upscale factor
        thickness = st.slider("Select line thickness", 0.5, 1.0, 0.5, step=0.01)
        upscale_factor = st.slider("Upscale factor (higher values reduce pixelation)", 1, 6, 2)
        
        processed_image, original_image = process_pixel_image(uploaded_file, thickness, upscale_factor)

        # Show original and processed images
        st.image([original_image, processed_image], caption=["Original Image", "Processed Image"], use_column_width=True)
        
        # Download options
        buf = BytesIO()
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        processed_image_pil.save(buf, format="PNG", dpi=(300, 300))

        st.download_button(
            label="Download Processed Image as PNG",
            data=buf.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )

    elif file_type == 'svg':
        # Handle vector-based images (SVG)
        st.write("Processing as an SVG image.")
        
        # Read SVG content
        svg_data = uploaded_file.read().decode()

        # Slider for adjusting stroke width
        new_stroke_width = st.slider("Adjust stroke width", 1, 10, 2)

        # Process the SVG by modifying stroke width
        modified_svg_data = process_svg_image(svg_data, new_stroke_width)

        # Display modified SVG as text
        st.text_area("Modified SVG Data", modified_svg_data, height=300)

        # Download the modified SVG
        st.download_button(
            label="Download Modified SVG",
            data=modified_svg_data,
            file_name="modified_image.svg",
            mime="image/svg+xml"
        )
