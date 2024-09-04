import cv2
import numpy as np
import streamlit
from PIL import Image
import matplotlib.pyplot as plt

# Function to process the image
def process_image(uploaded_image):
    # Convert the uploaded image to an OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Apply dilation to thicken the edges
    kernel = np.ones((2, 2), np.uint8)
    thickened_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a copy of the original image and apply the thickened black edges
    image_with_black_edges = image.copy()
    image_with_black_edges[thickened_edges != 0] = [0, 0, 0]  # Set thickened edges to black

    return image_with_black_edges, image  # Return the processed and original images

# Streamlit UI
st.title("Line Art Thickener")
st.write("Upload your line art and we'll thicken the edges for you!")

# Upload the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Process the image
    processed_image, original_image = process_image(uploaded_image)
    
    # Show both images side by side for comparison
    st.image([original_image, processed_image], caption=["Original Image", "Processed Image"], use_column_width=True)

    # Option to accept the processed image
    if st.button('Accept Processed Image'):
        st.success("You have accepted the processed image!")
        # Here you can add further functionality to save or download the image if needed
    else:
        st.warning("You haven't accepted the processed image yet.")
