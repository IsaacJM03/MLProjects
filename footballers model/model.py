import cv2                                 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk('../'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

x = os.path.join(BASE_DIR, "Images/Images/Group C/Argentina Players/Images_Lionel Messi (captain)")

messi_imgs=[]
dataset = x
for folder in os.listdir(dataset): 
    messi_imgs.append(folder)
len(messi_imgs)

y = os.path.join(BASE_DIR, "Images/Images/Group H/Portugal Players/Images_Cristiano Ronaldo (captain)")
ronaldo_imgs=[]
dataset = y
for folder in os.listdir(dataset): 
    ronaldo_imgs.append(folder)
len(ronaldo_imgs)

a=[]
for imag in messi_imgs:
    arr=cv2.imread(os.path.join(x,imag),cv2.IMREAD_GRAYSCALE)
    a.append(arr)

# Displaying some  images of messi
plt.figure(figsize=(10,10))
for i in range(len(a)):
    plt.subplot(10,5,i+1)
    plt.imshow(a[i],cmap='gray')

a2=[]
for image in ronaldo_imgs:
    arr2=cv2.imread(os.path.join(y,image),cv2.IMREAD_GRAYSCALE)
    a2.append(arr2)

# Displaying some  images of Ronaldo
plt.figure(figsize=(10,10))
for i in range(len(a2)):
    plt.subplot(10,5,i+1)
    plt.imshow(a2[i],cmap='gray')

# Resizing All images to (90,90) shape
IMG_SIZE=90

# Stroring all images with label 0,1
training_data=[]

def create_training_data():
    
    for imag in messi_imgs:
        arr=(os.path.join(x,imag))
        im_arr=cv2.imread(os.path.join(x,imag),cv2.IMREAD_GRAYSCALE)
        new_array=cv2.resize(im_arr,(IMG_SIZE,IMG_SIZE))
        training_data.append([new_array,0])
        
    for imag2 in ronaldo_imgs:
        arr=(os.path.join(y,imag2))
        im_arr2=cv2.imread(os.path.join(y,imag2),cv2.IMREAD_GRAYSCALE)
        new_array2=cv2.resize(im_arr2,(IMG_SIZE,IMG_SIZE))
        training_data.append([new_array2,1])

create_training_data()
len(training_data)

#Shuffling Data
random.shuffle(training_data)

#Checking top 5 labels
for sample in training_data[:5]:
    print(sample[1])
#Checking top 5 images
for sample in training_data[:5]:
    print(sample[0])

# Creating Features and labels
X=[]
y=[]
for features,label in training_data:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

#Checking the labels are they are balanced or not

label_messi,label_ronaldo=0,0

for i in y:
    if i==0:
        label_messi=label_messi+1
    else:
        label_ronaldo+=1
print(label_messi)
print(label_ronaldo)


dt = np.array([label_messi,label_ronaldo])
mylabels = ["label_messi",  "label_ronaldo"]
myexplode = [0, 0]
mycolors = [ "hotpink",  "#4CAF50"]
plt.pie(dt, labels = mylabels,explode=myexplode,colors=mycolors,startangle =90,
        autopct='%1.1f%%', shadow = True)
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show() 

def display_random_image( images, labels):
    
    
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if (labels[index==0]):
        plt.title('Image#{} : '.format(index) + "Messi")
    else:
        plt.title('Image#{} : '.format(index) + "Ronaldo")
    plt.show()


display_random_image( X, y)

def display_examples( images, labels):
    """
        Display 30 images from the images array with its corresponding labels
    """
    
    fig = plt.figure(figsize=(20,20))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(30):
        plt.subplot(5,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap="gray")
        plt.xlabel([labels[i]])
    plt.show() 

display_examples(X,y)

train_x=X/255.0
train_x=np.array(train_x)
y=np.array(y)

#Creating layers

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = train_x.shape[1:])) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Flatten())
model.add(Dense(64))
          
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history= model.fit(train_x ,y,batch_size=32,epochs=10, validation_split = 0.1)

def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_accuracy")
    plt.title("train_accuracy vs val_accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()

plot_accuracy_loss(history)