from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Create an image classification pipeline
model = pipeline("image-classification", model="IsaacMwesigwa/footballer-recognition-3")

# Define a route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle image predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image file from the client
        image_file = request.files['image']

        # Convert the file to a PIL Image
        pil_image = Image.open(image_file)

        # Make predictions using the pipeline
        results = model(pil_image)

        # Extract and return predictions
        predictions = results[0]
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
