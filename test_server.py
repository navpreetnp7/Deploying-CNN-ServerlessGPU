# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import banana_dev as banana

api_key = "79b191e0-8392-4e6b-bc52-f1440148e59a"
model_key = "2166cbfc-ce47-45e2-aa6f-bf87d6059230"

img_path = "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01440764_tench.jpeg"
model_inputs = {'input': img_path}

response = banana.run(api_key, model_key, model_inputs)
output = response['modelOutputs'][0]['output']

print(output)