from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleNN 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode

app = Flask(__name__)

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))
# model.eval()

# Define transformations for the uploaded image
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

# Initialize lists to store evaluation metrics
uploaded_data_metrics = []
expected_values = []
predicted_values = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global transform,uploaded_data_metrics,predicted_values,expected_values
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Open the image file
            img = Image.open(uploaded_file)

            # Resize the image to 28x28 pixels
            img = img.resize((28, 28))

            # Convert to grayscale
            img = img.convert('L')

            # Transform the image for the model
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Transform the image for the model
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Make the prediction using the model
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted_class = torch.max(output, 1)

            # Display the predicted class
            predicted_digit = predicted_class.item()

            print(f"Predicted digit: {predicted_digit}")

            expected_digit = int(request.form['expected_digit'])
            
            # store the expected and predicted values
            expected_values.append(expected_digit)
            predicted_values.append(predicted_digit)

            # Check if the prediction is correct
            correct_prediction = expected_digit == predicted_digit
            accuracy = sum(uploaded_data_metrics) / len(uploaded_data_metrics) if uploaded_data_metrics else 0

            # Store the evaluation metric
            uploaded_data_metrics.append(correct_prediction)

            # Plot the evaluation metrics over time
            plot_evaluation_metrics()

            return render_template('index.html', prediction=predicted_digit,accuracy=accuracy)

    return render_template('index.html', prediction=None,accuracy=0)

def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def make_prediction(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    
def plot_evaluation_metrics():
    global uploaded_data_metrics,expected_values,predicted_values

    # Plot expected and predicted values
    plt.subplot(1, 2, 1)
    plt.plot(expected_values, label='Expected', marker='o', linestyle='--', color='blue')
    plt.plot(predicted_values, label='Predicted', marker='x', linestyle='-', color='red')
    plt.title('Expected vs Predicted Values')
    plt.xlabel('Data Points')
    plt.ylabel('Digit')
    plt.legend()

    # Plot accuracy over time
    plt.subplot(1, 2, 2)
    plt.plot(uploaded_data_metrics, label='Uploaded Data Accuracy', color='green')
    plt.title('Uploaded Data Accuracy')
    plt.xlabel('Data Points')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics.png')  # Save the plot as an image
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
