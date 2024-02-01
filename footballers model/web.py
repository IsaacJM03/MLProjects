from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from model_pytorch import CNNModel  # Assuming your model is defined in a separate file

app = Flask(__name__)

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((90, 90)),
    transforms.ToTensor(),
])

# Initialize the model
model = CNNModel()
model.load_state_dict(torch.load('football_model.pth'))
model.eval()

# def predict_player(player_name, test_images_folder):
#     results = []

#     # Get the specific player's folder
#     player_folder = os.path.join(player_folder_path, "Images", f"Group {group_name}", f"{country_name} Players", f"Images_{player_name}")

#     if not os.path.exists(player_folder):
#         return [{"filename": "Error", "predicted_player": f"Folder for {player_name} not found.", "probability": 0.0}]

#     # Loop through test images
#     for filename in os.listdir(test_images_folder):
#         if filename.endswith((".jpg", ".png", ".jpeg", ".webp")):
#             # Read and preprocess the image
#             img_path = os.path.join(test_images_folder, filename)
#             img = Image.open(img_path)
#             img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

#             # Make the prediction
#             with torch.no_grad():
#                 prediction = model(img_tensor).item()
#                 probability = torch.sigmoid(torch.Tensor([prediction])).item()

#             # Interpret the model's output
#             predicted_player = player_name if float(prediction) >= float(0.5) else "Not " + player_name

#             results.append({
#                 "filename": filename,
#                 "predicted_player": predicted_player,
#                 "probability": probability * 100
#             })

#     return results

def predict_player(player_name, group_name, country_name, player_folder_path, test_images_folder):
    results = []

    # Get the specific player's folder
    player_folder = os.path.join(player_folder_path, "Images", f"Group {group_name}", f"{country_name} Players", f"Images_{player_name}")

    if not os.path.exists(player_folder):
        return [{"filename": "Error", "predicted_player": f"Folder for {player_name} not found.", "probability": 0.0}]

    # Loop through test images
    for filename in os.listdir(test_images_folder):
        if filename.endswith((".jpg", ".png", ".jpeg", ".webp")):
            # Read and preprocess the image
            img_path = os.path.join(test_images_folder, filename)
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

            # Make the prediction
            with torch.no_grad():
                prediction = model(img_tensor).item()
                probability = torch.sigmoid(torch.Tensor([prediction])).item()

            # Interpret the model's output
            predicted_player = player_name if float(prediction) >= float(0.5) else "Not " + player_name

            results.append({
                "filename": filename,
                "predicted_player": predicted_player,
                "probability": probability * 100
            })

    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        player_name = request.form['player_name']
        group_name = request.form['group_name']
        country_name = request.form['country_name']
        test_folder_path = request.form['test_folder_path']

        results = predict_player(player_name, group_name, country_name, test_folder_path)
        return render_template('result.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
