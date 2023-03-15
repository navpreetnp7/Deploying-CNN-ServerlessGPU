import torch
from model.model import OnnxModel
from PIL import Image
import time
import requests
from io import BytesIO


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    global model
    path = "model/classifier.onnx"
    model = OnnxModel(path)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    img_path = model_inputs.get('input', None)
    response = requests.get(img_path)
    img = Image.open(BytesIO(response.content))

    if img == None:
        return {'message': "No image provided"}
    
    # Run the model
    result = model.predict(img)

    # Return the results as a dictionary
    return {'output' : torch.argmax(torch.Tensor(result))}
