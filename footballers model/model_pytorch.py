import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

IMG_SIZE = 90
# Define a simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        IMG_SIZE = 90
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        # Basically, the sigmoid function determines which value to pass as output and what not to pass as output. Removes non-linearity and produces a probability
        x = self.sigmoid(self.fc2(x))
        return x

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img) if self.transform else img
        return img, label

# Plot accuracy and loss
def plot_accuracy_loss(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation losses
    plt.subplot(221)
    plt.plot(epochs, train_losses, 'bo--', label="train_loss")
    plt.plot(epochs, val_losses, 'ro--', label="val_loss")
    plt.title("Training and Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(222)
    plt.plot(epochs, train_accuracies, 'bo--', label="train_acc")
    plt.plot(epochs, val_accuracies, 'ro--', label="val_acc")
    plt.title("Training and Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


# Load folders and files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
x_folder = os.path.join(BASE_DIR, "Images/Images/Group C/Argentina Players/Images_Lionel Messi (captain)")
y_folder = os.path.join(BASE_DIR, "Images/Images/Group H/Portugal Players/Images_Cristiano Ronaldo (captain)")

# Load images and labels
messi_imgs = [(cv2.imread(os.path.join(x_folder, img), cv2.IMREAD_GRAYSCALE), 0) for img in os.listdir(x_folder)]
ronaldo_imgs = [(cv2.imread(os.path.join(y_folder, img), cv2.IMREAD_GRAYSCALE), 1) for img in os.listdir(y_folder)]

# Combine Messi and Ronaldo data
training_data = messi_imgs + ronaldo_imgs

# Shuffle data
random.shuffle(training_data)

# Resize images to a common size (e.g., 90x90)
resized_images = [(cv2.resize(img, (90, 90)), label) for img, label in training_data]

# Split the data into images and labels
X = np.array([item[0] for item in resized_images])
y = np.array([item[1] for item in resized_images])
# Reshape X and y
X = X.reshape(-1, IMG_SIZE, IMG_SIZE)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Create datasets and dataloaders
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image
    transforms.RandomRotation(10),            # Rotate by a random angle up to 10 degrees
    transforms.RandomHorizontalFlip(),        # Randomly flip horizontally
    # transforms.Resize((90, 90)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


train_dataset = CustomDataset(list(zip(torch.from_numpy(X_train).unsqueeze(1), y_train)), transform=transform)
val_dataset = CustomDataset(list(zip(torch.from_numpy(X_val).unsqueeze(1), y_val)), transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)  # Add weight decay parameter

# Training loop
num_epochs = 40
train_losses = []  # To store training losses
val_losses = []    # To store validation losses
train_accuracies = []  # To store training accuracies
val_accuracies = []    # To store validation accuracies
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        predicted = (outputs >= 0.5).float()
        correct_train += (predicted == labels.view(-1, 1)).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_outputs = []
        val_labels = []
        for val_images, val_labels_batch in val_loader:
            val_outputs_batch = model(val_images.float())
            val_outputs.extend(val_outputs_batch.cpu().numpy())
            val_labels.extend(val_labels_batch.numpy())

    val_outputs = np.array(val_outputs)
    val_labels = np.array(val_labels)
    val_loss = criterion(torch.Tensor(val_outputs), torch.Tensor(val_labels).view(-1, 1))

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    val_accuracy = (np.round(val_outputs) == val_labels.reshape(-1, 1)).mean()
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Training Acc: {train_accuracy:.4f}, Validation Loss: {val_loss.item():.4f}')
# plot_accuracy_loss(train_losses, val_losses, train_accuracies, val_accuracies)

torch.save(model.state_dict(), 'football_model.pth')

