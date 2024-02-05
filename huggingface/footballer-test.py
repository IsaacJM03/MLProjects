import os
from PIL import Image
import numpy as np
from transformers import AutoFeatureExtractor, ViTForImageClassification, ViTConfig, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the environment variable
path = os.getenv("IMAGE_PATH")
# Path to the root directory of your dataset
root_path = path

# Function to load dataset
def load_custom_dataset(root_path):
    data = []
    labels = set()
    
    for player_folder in os.listdir(root_path):
        player_path = os.path.join(root_path, player_folder)
        
        if os.path.isdir(player_path):
            image_paths = [os.path.join(player_path, image_file) for image_file in os.listdir(player_path)
                           if image_file.lower().endswith((".png", ".jpg", ".jpeg"))]
            
            if image_paths:
                data.append({"images": image_paths, "label": player_folder})
                labels.add(player_folder)
    
    return data, list(labels)

# Load dataset
all_players_data, all_players_labels = load_custom_dataset(root_path)

# Split into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    all_players_data, all_players_labels, test_size=0.2, random_state=42
)

# Create a dictionary containing train set only
custom_dataset = {"train": train_data}

# Save the dataset using the datasets library
dataset_folder = "custom_dataset"
# os.makedirs(dataset_folder, exist_ok=True)

# for split, data in custom_dataset.items():
#     split_folder = os.path.join(dataset_folder, split)
#     os.makedirs(split_folder, exist_ok=True)

#     for i, example in enumerate(data):
#         label_folder = os.path.join(split_folder, f"{i}_{example['label']}")
#         os.makedirs(label_folder, exist_ok=True)

#         for j, image_path in enumerate(example["images"]):
#             image = Image.open(image_path).convert("RGB")
#             image.save(os.path.join(label_folder, f"{j}.png"))

# Use AutoFeatureExtractor for tokenization
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

def tokenize_function(example):
    # Load the images from the list of paths
    images = [Image.open(image_path).convert("RGB") for image_path in example["images"]]
    
    # Convert PIL Images to NumPy arrays
    images_np = [np.array(image) for image in images]
    
    # Tokenize the images
    return feature_extractor(images=images_np, return_tensors="pt")

# Tokenize the dataset
tokenized_datasets = custom_dataset
tokenized_datasets["train"] = [tokenize_function(example) for example in train_data]

# Initialize the model and training arguments
model_name = "google/vit-base-patch16-224"
config = ViTConfig.from_pretrained(model_name, num_labels=831)

model = ViTForImageClassification(config)

training_args = TrainingArguments(
    output_dir="./vit_finetuned",
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    logging_steps=50,
)

# Create Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
trainer.save_model("./footballer-model")
trainer.save_model("./footballer-model.pth")
 
