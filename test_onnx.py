import torch
from PIL import Image
from model.model import OnnxModel
from io import BytesIO
import requests

global model
path = "model/classifier.onnx"
model = OnnxModel(path)

class TestOnnx:

    def test_tench(self):
        global model
        img_path = "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01440764_tench.jpeg"
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
        res = model.predict(img)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([0])

    def test_turtle(self):
        global model
        img_path = "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01667114_mud_turtle.JPEG"
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
        res = model.predict(img)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([35])