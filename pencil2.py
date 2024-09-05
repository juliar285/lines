import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Load the CycleGAN or Pix2Pix model from PyTorch Hub
@st.cache(allow_output_mutation=True)
def load_model():
    # For Pix2Pix or CycleGAN, we need to specify the task and use 'pretrained=True'
    model = torch.hub.load('junyanz/pytorch-CycleGAN-and-pix2pix', 'pix2pix', pretrained=True, model='facades')
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
        output = model(img_tensor)  # Updated call to match the new model
        sketch_image = output[0]  # Taking the first element if multiple are returned
    
    # Convert the tensor to an image
    sketch_image = (sketch_image.squeeze().permute(1, 2, 0) + 1) / 2  # Denormalize to [0, 1] range
    sketch_image = sketch_image.numpy()

    # Display the original and generated sketch
    st.image([img, sketch_image], caption=['Original Image', 'Generated Sketch'], use_column_width=True)
