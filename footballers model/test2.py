import torch
from torchvision import transforms
from PIL import Image
import os
from model_pytorch import CNNModel  # Assuming your model is defined in a separate file

def test_model_for_player(player_name, group_name, country_name, player_images_folder, test_images_folder):
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

    # Get the specific player's folder
    player_folder = os.path.join(player_images_folder, "Images", f"Group {group_name}", f"{country_name} Players", f"Images_{player_name}")
    print(player_folder)
    if not os.path.exists(player_folder):
        print(f"Error: Folder for {player_name} not found.")
        return

    # Loop through test images
    for filename in os.listdir(test_images_folder):
        if filename.endswith((".jpg", ".png", ".jpeg", ".webp")):  # Assuming all images are in jpg format
            # Read and preprocess the image
            img_path = os.path.join(test_images_folder, filename)
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0)  # Add a batch dimension

            # Make the prediction
            with torch.no_grad():
                prediction = model(img_tensor).item()
                probability = torch.sigmoid(torch.Tensor([prediction])).item()
            
            # Interpret the model's output
            predicted_player = player_name if prediction > 0.5 else "Not " + player_name

            print(f"Image: {filename}, Predicted Player: {predicted_player}, Probability: {probability * 100:.2f}%")

# Specify the player, group, country, and paths for player and test folders
player_to_test = "Cristiano Ronaldo (captain)"
group_to_test = "H"
country_to_test = "Portugal"
player_folder_path = "/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/Images"
test_folder_path = "/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/test"
# test_folder_path = "/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/Images/Images/Group H/Portugal Players/Images_Cristiano Ronaldo (captain)"
# Call the function to test the model for the specified player
test_model_for_player(player_to_test, group_to_test, country_to_test, player_folder_path, test_folder_path)
