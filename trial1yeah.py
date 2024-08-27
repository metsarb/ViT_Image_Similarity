import ssl
import certifi
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

#Adding ssl certificate blocker because of using MacOS
#Calling libraries transformers, pillow, sklearn, torch

device = torch.device('cpu')

#Choosing the device because of embedded GPU of MacBook is not efficient to use in this project

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

#Defining the processor to fit the image's data to the model's need and AI model for the computer and compiler

image_path = "/---/---/---/---/---/---"

#Defining the image path to use it in the loops and functions

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images = image, return_tensors = "pt").to(device)
    return inputs

#Creating a function to define that calculation will be on RGB format using PyTorch

def get_embedding(image_path):
    inputs = process_image(image_path)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.logits
        return embeddings

#Making calculations on CPU and taking inputs as image matrices
#process_image calls the image and compute the data in the needed tensor format
#Using no_grad to disable derivative operations

image1_path = "/---/---/---/---/---/---"
image2_path = "/---/---/---/---/---/---"

#Defining image paths to compare

embedding1 = get_embedding(image1_path)
embedding2 = get_embedding(image2_path)

#Embedding the images

def compare_embeddings(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

#Creating compare function to calculate the similarity percentage

results1 = compare_embeddings(embedding1, embedding2)
results2 = compare_embeddings(embedding2, embedding1)


similarity = compare_embeddings(embedding1.numpy(), embedding2.numpy())
print(f"Similarity Score: {similarity}")

#Computing similarity percentage with numpy library and printing
