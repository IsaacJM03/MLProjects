import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from model_pytorch import CNNModel  # Assuming your model is defined in a separate file

# Initialize the model
model = CNNModel()
model.load_state_dict(torch.load('football_model.pth'))
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((90, 90)),
    transforms.ToTensor(),
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the root folder containing test images
test_folder = os.path.join(BASE_DIR, "Images/Images/Group C/Argentina Players/Images_Lionel Messi (captain)")

# Loop through test images
for filename in os.listdir(test_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".webp"):  # Assuming all images are in jpg format
        # Read and preprocess the image
        img_path = os.path.join(test_folder, filename)
        img = Image.open(img_path)
        img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

        # Make the prediction
        with torch.no_grad():
            prediction = model(img_tensor).item()
            probability = torch.sigmoid(torch.Tensor([prediction])).item()
            # print(f"Prediction: {prediction * 100:.2f}%")
        # Interpret the model's output
        predicted_player = "Lionel Messi" if prediction > 0.5 else "Not Messi"
        
        print(f"Image: {filename}, Predicted Player: {predicted_player}, Probability: {probability * 100:.2f}%")
