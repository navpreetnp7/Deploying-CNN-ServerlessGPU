from model.pytorch_model import Classifier, BasicBlock

import io
import numpy as np

from torch import nn
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import onnx


def convert_to_onnx(model_url, path, batch_size, image_size):

    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_url))
    model.eval()

    x = torch.randn(batch_size, image_size[0], image_size[1], image_size[2], requires_grad=True)

    torch.onnx.export(model,               
                    x,                         
                    path,   
                    export_params=True,       
                    opset_version=10,        
                    do_constant_folding=True,  
                    input_names = ['input'], 
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'},  
                                    'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    model_url = 'model/pytorch_model_weights.pth'
    path = "model/classifier.onnx"
    batch_size = 1
    image_size = (3, 224, 224)

    convert_to_onnx(model_url, path, batch_size, image_size)

    onnx_model = onnx.load(path)

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")

