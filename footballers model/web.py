# Modified Flask app code
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model_pytorch import CNNModel 
import os

app = Flask(__name__)

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
# Define the root folder containing player folders
root_folder = os.path.join(BASE_DIR, "Images/Images")

def predict_class(input_img_tensor):
    """
    Predict the class (football player) using the trained model.
    """
    with torch.no_grad():
        # Ensure the model is in evaluation mode
        model.eval()
        
        # Forward pass
        output = model(input_img_tensor.unsqueeze(0))

        # Get the predicted class (index)
        predicted_class = torch.argmax(output).item()

    return predicted_class

def find_best_match(input_img_tensor, selected_player):
    """
    Find the best match across all player image folders using the trained model.
    """
    best_match = None
    best_player = None

    # Iterate through different folders
    for group_folder in os.listdir(root_folder):
        group_folder_path = os.path.join(root_folder, group_folder)

        if os.path.isdir(group_folder_path):  # Ensure it's a directory
            for country_folder in os.listdir(group_folder_path):
                country_folder_path = os.path.join(group_folder_path, country_folder)

                if os.path.isdir(country_folder_path):  # Ensure it's a directory
                    for player_parent_folder in os.listdir(country_folder_path):
                        player_parent_folder_path = os.path.join(country_folder_path, player_parent_folder)

                        # Check if the folder contains "Images_" and is a directory
                        if "Images_" in player_parent_folder and os.path.isdir(player_parent_folder_path):
                            player_name = player_parent_folder.split("_")[-1]  # Extract the player's name

                            # Check if the current player matches the selected player
                            if selected_player.lower() == player_name.lower():
                                # Get the image file path for the current player
                                image_files = [f for f in os.listdir(player_parent_folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                                if image_files:
                                    image_path = os.path.join(player_parent_folder_path, image_files[0])  # Use the first image

                                    # Read and preprocess the player image
                                    player_img = Image.open(image_path)
                                    player_img_tensor = transform(player_img).unsqueeze(0)

                                    # Predict the class for the player image
                                    player_class = predict_class(player_img_tensor)

                                    # Predict the class for the uploaded image
                                    uploaded_class = predict_class(input_img_tensor)

                                    # Compare predicted classes
                                    if player_class == uploaded_class:
                                        best_match = image_files[0]  # Update with the actual filename
                                        best_player = player_name

    return best_player, best_match

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file and selected player from the request
        uploaded_file = request.files['file']
        selected_player = request.form['player']

        if uploaded_file.filename != '':
            # Ensure the file is saved
            uploaded_file_path = os.path.join(BASE_DIR, 'uploaded_image.jpg')
            uploaded_file.save(uploaded_file_path)
            # Read and preprocess the image
            img = Image.open(uploaded_file).convert('L')
            img_tensor = transform(img)

            # Find the best match for the selected player across all player folders
            player_name, best_match = find_best_match(img_tensor, selected_player)

            return render_template('result.html', result=player_name, best_match=best_match)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
