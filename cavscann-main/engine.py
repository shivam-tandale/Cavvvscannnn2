import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import streamlit as st
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pretrained EfficientNet-B0 model
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Modify the classifier for 6 classes
num_classes = 6
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

# Load the model checkpoint with strict=False
model_checkpoint_path = 'Model_Assets/best_model.pth'
try:
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print("Model checkpoint loaded successfully.")
except Exception as e:
    print(f"Error loading model checkpoint: {e}")

model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def prediction(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Softmax for probabilities
            index = torch.argmax(probabilities).item()
            confidence = probabilities[index].item()
        st.write(f"Prediction: {index}, Confidence: {confidence}")

        return index, confidence

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None