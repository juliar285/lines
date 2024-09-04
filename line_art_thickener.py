import streamlit as st
import svgwrite
import potrace
from PIL import Image
import numpy as np
from io import BytesIO

# Function to convert PNG/JPG to black-and-white bitmap for Potrace
def convert_image_to_bw(uploaded_image):
    # Open the image using PIL
    image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize the image
    return image

# Function to convert the BW image to SVG using Potrace
def convert_to_svg(bw_image):
    # Convert image to numpy array and prepare for Potrace
    bitmap = potrace.Bitmap(np.array(bw_image))
    path = bitmap.trace()

    # Create an SVG document
    dwg = svgwrite.Drawing(size=(bw_image.width, bw_image.height))
    
    # Add paths from Potrace to SVG
    for curve in path:
        d = "M {} {} ".format(curve.start_point[0], curve.start_point[1])
        for segment in curve:
            if isinstance(segment, potrace.Curve.Corner):
                d += "L {} {} ".format(segment.c[0], segment.c[1])
            else:
                d += "C {} {} {} {} {} {} ".format(segment.c1[0], segment.c1[1], segment.c2[0], segment.c2[1], segment.end_point[0], segment.end_point[1])
        d += "Z"
        dwg.add(dwg.path(d, fill="none", stroke="black", stroke_width="1"))

    return dwg.tostring()

# Function to modify the stroke width of the generated SVG
def modify_svg_stroke(svg_data, stroke_width=3):
    dwg = svgwrite.Drawing()
    for elem in svgwrite.etree.fromstring(svg_data).getchildren():
        if elem.tag.endswith('path'):
            elem.set('stroke-width', str(stroke_width))
        dwg.add(dwg.element_fromstring(svgwrite.etree.tostring(elem)))
    return dwg.tostring()

# Streamlit UI
st.title("Image to SVG Converter with Line Thickening")
st.write("Upload your PNG or JPG, we'll convert it to SVG, and allow you to thicken the lines.")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Convert uploaded image to black-and-white
    bw_image = convert_image_to_bw(uploaded_image)
    
    # Convert the BW image to SVG using Potrace
    svg_data = convert_to_svg(bw_image)
    
    # Modify the stroke width
    stroke_width = st.slider("Adjust line thickness:", 1, 10, 3)
    modified_svg = modify_svg_stroke(svg_data, stroke_width)
    
    # Show original image and provide SVG download link
    st.image(uploaded_image, caption="Original Image")
    st.download_button(label="Download Modified SVG", data=modified_svg, file_name="modified_image.svg", mime="image/svg+xml")
