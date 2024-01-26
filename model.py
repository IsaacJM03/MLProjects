import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# choose a dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)


# define a model
class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN,self).__init__()
    self.fc = nn.Linear(28*28,10)

  def forward(self,x):
    x = x.view(-1,28*28)
    x = self.fc(x)
    return x
  
model = SimpleNN()

# define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01) #learning rate of 0.01

# train the model
num_epochs = 5
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
  # Track training loss and accuracy for each epoch
  training_loss = 0.0
  correct_train = 0
  total_train = 0
  for inputs,labels in train_loader:
    optimizer.zero_grad()

    # Print the input data (images) and labels
    # print("Input Data:", inputs)
    print("Labels:", labels)
    outputs = model(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    training_loss += loss.item()

    # Update training accuracy
    _, predicted_train = torch.max(outputs.data, 1)
    total_train += labels.size(0)
    correct_train += (predicted_train == labels).sum().item()
  # Calculate and store average training loss and accuracy for the epoch
  avg_train_loss = training_loss / len(train_loader)
  avg_train_accuracy = correct_train / total_train

  train_losses.append(avg_train_loss)
  train_accuracies.append(avg_train_accuracy)

  # Print training results for the epoch
  print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_accuracy:.4f}")

torch.save(model.state_dict(), 'mnist_model.pth')
# test the model
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)

# correct = 0
# total = 0

# with torch.no_grad():
#   for inputs,labels in test_loader:
#     outputs = model(inputs)
#     _,predicted = torch.max(outputs.data,1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print('Accuracy of the model on the test images: {:.2%}'.format(accuracy))


# # Plotting
# plt.figure(figsize=(10, 5))

# # Plot training and testing loss
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Training Loss', color='blue')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Plot training accuracy
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Training Accuracy', color='green')
# plt.title('Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# '''
# Training loss is a measure of how well the model is performing on the training data during the training process. It quantifies the difference between the predicted output and the actual target values for the training set.
# The goal during training is to minimize the training loss. Lower values of the training loss indicate that the model is getting closer to making accurate predictions on the training data.


# Training accuracy is a measure of how many training examples are correctly classified by the model during training.
# It is calculated by comparing the predicted class labels to the actual class labels for the training set.


# epoch is like each iteration through the data set.
# '''