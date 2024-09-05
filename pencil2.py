import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Load pre-trained Pix2Pix model from an alternative source
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the model from an alternative repository
    repo_or_dir = 'https://github.com/phillipi/pix2pix'
    model = torch.hub.load(repo_or_dir, 'pix2pix', pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define image preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = preprocess(image).unsqueeze(0)
    return img_tensor

# Streamlit app layout
st.title('Pix2Pix Sketch Generation')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess the image
    img_tensor = preprocess_image(img)
    
    # Generate the sketch using Pix2Pix
    with torch.no_grad():
        sketch_image = model.inference(img_tensor)
    
    # Convert the tensor to an image
    sketch_image = (sketch_image.squeeze().permute(1, 2, 0) + 1) / 2  # Denormalize to [0, 1] range
    sketch_image = sketch_image.numpy()

    # Display the original and generated sketch
    st.image([img, sketch_image], caption=['Original Image', 'Generated Sketch'], use_column_width=True)
