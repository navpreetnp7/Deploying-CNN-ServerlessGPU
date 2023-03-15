import torch
from PIL import Image
from model.model import OnnxModel

global model
path = "model/classifier.onnx"
model = OnnxModel(path)

class TestOnnx:

    def test_tench(self):
        global model
        img = Image.open("images/n01440764_tench.JPEG")
        res = model.predict(img)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([0])

    def test_turtle(self):
        global model
        img = Image.open("images/n01667114_mud_turtle.JPEG")
        res = model.predict(img)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([35])