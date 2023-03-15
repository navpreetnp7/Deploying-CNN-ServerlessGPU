import torch
from PIL import Image
from model import OnnxModel

class TestOnnx:

    def test_tench(self):
        path = "model/classifier.onnx"
        model = OnnxModel(path)

        img = Image.open("images/n01440764_tench.JPEG")
        inp = model.preprocess_numpy(img).unsqueeze(0) 
        res = model.predict(inp)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([0])

    def test_turtle(self):
        path = "model/classifier.onnx"
        model = OnnxModel(path)

        img = Image.open("images/n01667114_mud_turtle.JPEG")
        inp = model.preprocess_numpy(img).unsqueeze(0) 
        res = model.predict(inp)
        assert torch.argmax(torch.Tensor(res)) == torch.Tensor([35])