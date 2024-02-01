# token = "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"

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id)

# text = "Hello my name is"
# inputs = tokenizer(text, return_tensors="pt")

# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# import requests
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
# API_TOKEN = "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"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# def query(payload):
#     data = {"inputs": payload}
#     response = requests.post(API_URL, headers=headers, json=data)
#     return response.json()

# text_to_infer = "Hello my name is"
# data = query(text_to_infer)
# print(data)


# import requests

# API_URL = "https://api-inference.huggingface.co/models/impira/layoutlm-document-qa"
# headers = {"Authorization": "Bearer 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"}

# import base64

# def process_image(payload):
#     print(payload)
#     with open(payload["inputs"]["image"], "rb") as file:
#         image_data = file.read()
#         payload["inputs"]["image"] = base64.b64encode(image_data).decode("utf-8")
#     response = requests.post(API_URL,headers=headers, json=payload)
#     return response.json()

# output = process_image({
#     "inputs": {
# 		"image": "/home/isaac-flt/Downloads/problemset1.jpeg",
# 		"question": "help me answer these questions"
# 	},
# })

# print(output)


import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the environment variable
token = os.getenv("HUGGINGFACE_API_KEY")
# API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# headers = {"Authorization": f"Bearer {token} "}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content
# image_bytes = query({
# 	"inputs": "Elevation worship",
# })
# # You can access the image with PIL.Image for example
# import io
# from PIL import Image
# image = Image.open(io.BytesIO(image_bytes))
# Image._show(image)


# API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": {
# 		"past_user_inputs": ["Which movie is the best ?"],
# 		"generated_responses": ["It is The Matrix for sure."],
# 		"text": "Can you explain the matrix ?",
# 		"padding_side": "left"
# 	},
# })

# print(output)

import requests

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"
headers = {"Authorization": f"Bearer {token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "def test_model_for_player(player_name, group_name, country_name, player_images_folder, test_images_folder):",
})
print(output)