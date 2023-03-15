import numpy as np
import onnxruntime as ort

import torch
from torchvision import transforms
from PIL import Image

class OnnxModel:

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_data):
        input_data = self.preprocess_numpy(input_data).unsqueeze(0)
        input_data = np.array(input_data)
        input_data = input_data.astype(np.float32)
        output = self.session.run([self.output_name], {self.input_name: input_data})
        return output[0]

    def preprocess_numpy(self, img):
        resize = transforms.Resize((224, 224))  
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img
    
if __name__ == "__main__":

    path = "model/classifier.onnx"
    model = OnnxModel(path)

    