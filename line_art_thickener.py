import streamlit as st
import cairosvg
import svgwrite
from io import BytesIO
from PIL import Image
import numpy as np

# Function to convert the uploaded PNG image to SVG
def convert_image_to_svg(uploaded_image):
    # Read the uploaded image
    image = Image.open(uploaded_image)
    
    # Save the image to a byte stream
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Convert PNG to SVG using cairosvg
    svg_output = cairosvg.svg2svg(bytestring=image_bytes.read())
    
    return svg_output

# Function to modify the SVG paths and increase the stroke width
def modify_svg_stroke(svg_data, stroke_width=3):
    # Create an SVG document using svgwrite
    dwg = svgwrite.Drawing()
    # Parse the original SVG and update stroke-width for all paths
    for elem in svgwrite.etree.fromstring(svg_data).getchildren():
        if elem.tag.endswith('path'):  # Modify only paths
            elem.set('stroke-width', str(stroke_width))  # Modify stroke-width
        dwg.add(dwg.element_fromstring(svgwrite.etree.tostring(elem)))

    # Save modified SVG to a string
    return dwg.tostring()

# Streamlit UI
st.title("Line Art SVG Manipulator")
st.write("Upload your line art, and we'll convert it to SVG, thicken the lines, and display it.")

# Upload the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Convert image to SVG
    svg_data = convert_image_to_svg(uploaded_image)
    
    # Modify the stroke width in the SVG paths
    stroke_width = st.slider("Adjust line thickness:", 1, 10, 3)  # Slider for line thickness
    modified_svg = modify_svg_stroke(svg_data, stroke_width)
    
    # Display the original and modified SVG
    st.write("Here is your processed SVG with thicker lines:")
    st.image(uploaded_image, caption="Original Image")
    
    # Display the modified SVG (as SVGs are scalable, they may need specific rendering in Streamlit)
    st.download_button(label="Download Modified SVG", data=modified_svg, file_name="modified_image.svg", mime="image/svg+xml")
