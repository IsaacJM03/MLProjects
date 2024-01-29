from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Initialize the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
    transforms.Resize((90, 90)),
    transforms.ToTensor(),
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the root folder containing player folders
root_folder = os.path.join(BASE_DIR, "Images/Images")

# Initialize an empty dictionary to store mappings between images and players
image_player_mapping = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Ensure the file is saved
            uploaded_file_path = os.path.join(BASE_DIR, 'uploaded_image.jpg')
            uploaded_file.save(uploaded_file_path)
            # Read and preprocess the image
            img = Image.open(uploaded_file)
            img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

            # Make the prediction
            with torch.no_grad():
                prediction = model(img_tensor)

            # Convert prediction to player name (replace this with your actual mapping logic)
            player_name = "Ronaldo"

            # Associate the image with the player name
            image_player_mapping[uploaded_file.filename] = player_name

            return render_template('result.html', result=player_name)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
