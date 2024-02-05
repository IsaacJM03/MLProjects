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

# API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "def test_model_for_player(player_name, group_name, country_name, player_images_folder, test_images_folder):",
# })
# print(output)

# import requests

# API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
# headers = {"Authorization": f"Bearer {token}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "when it rains, ",
# 	"options": {
# 		"length_penalty": 2.5,
# 		"padding_side" : "left"
# 	}
# })
# # output.padding_side = "left"

# print(output)


# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")

# print(pipe)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)


tokenizer = AutoTokenizer.from_pretrained("./huggingface/dialogpt-tokenizer/")
model = AutoModelForCausalLM.from_pretrained("./huggingface/dialogpt-model/")

# tokenizer.save_pretrained("./huggingface/tokenizers/")
# model.save_pretrained("./huggingface/models/")

message = input("how many messages do you wanna send?: ")
# Let's chat for some lines
for step in range(int(message)):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


# from transformers import pipeline

# image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# result = image_to_text("/home/isaac-flt/Projects/ML4D/MLProjects/footballers model/test/charge-messi-4k-ultra-hd-j29dpoy6dmug491j.webp")
# print(result)

# # [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
