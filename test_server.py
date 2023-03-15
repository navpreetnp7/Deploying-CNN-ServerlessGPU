# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import banana_dev as banana
import argparse
import torch
from PIL import Image
import requests
from io import BytesIO

api_key = "79b191e0-8392-4e6b-bc52-f1440148e59a"
model_key = "9101352f-830c-4dc6-9f48-f12092609d42"

def test_tench(img_path):
    img_path = "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01440764_tench.jpeg"
    output, time = get_prediction(img_path)
    assert output == 0
    print('Image is correctly predicted to belong to class tench')

def test_turtle(img_path):
    img_path = "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01667114_mud_turtle.JPEG"
    output, time = get_prediction(img_path)
    assert output == 35
    print('Image is correctly predicted to belong to class mud_turtle')

def get_prediction(img_path):

    model_inputs = {'input': img_path}
    response = banana.run(api_key, model_key, model_inputs)
    output = response['modelOutputs'][0]['output']
    time = response['modelOutputs'][0]['time']

    return output, time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='input the path of the image to be predicted')
    parser.add_argument('--test', action='store_true', help='run preset custom tests')

    args = parser.parse_args()
    if args.test:
        test_tench()
        test_turtle()

    output, time = get_prediction(args.image_path)
    print(f'The image belongs to class id: {output}')
    print(f'Time taken for one banana dev call: {time:.2f} seconds\n')
    
