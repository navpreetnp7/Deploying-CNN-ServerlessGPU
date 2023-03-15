# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import banana_dev as banana

api_key = "79b191e0-8392-4e6b-bc52-f1440148e59a"
model_key = "4ef22b5a-a258-4eb7-ae78-ca456c8d111e"

img_path = "images/n01440764_tench.JPEG"
model_inputs = {'input': img_path}

response = banana.run(api_key, model_key, model_inputs)
output = response['modelOutputs'][0]['output']

print(output)