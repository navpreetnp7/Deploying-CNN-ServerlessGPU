# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
from PIL import Image

api_url = 'http://localhost:8000/'

img = Image.open("images/n01440764_tench.JPEG")
model_inputs = {'input': img.tolist()}

response = requests.post(api_url, json = model_inputs)

print(response.json())