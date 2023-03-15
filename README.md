
# 🍌 Deploying Classification Neural Network on Serverless GPU platform of Banana Dev
This project provides a codebase to convert a PyTorch model to an ONNX model and deploy it to a banana dev platform.



## Installation

To run the scripts, simply clone this repo and install all the dependencies

```
git clone https://github.com/navpreetnp7/Deploying-CNN-ServerlessGPU.git
pip install -r requirements.txt
```



## Deliverables

#####  `1. convert_to_onnx.py`

Simply run the script which would export the pytorch model to ONNX

```
python convert_to_onnx.py
```

##### `2. model.py`

This file contains a class to instantiate the model from the ONNX file and then run inference using ONNX runtime.

The 'preprocess_numpy' function is used to preprocess the given image by apply normalisation and standardisation.

The 'predict' method gives the class output of the given image

#####  `3. test_onnx.py`

The script tests the converted ONNX model on CPU. It uses two  images from the repository and verifies if the model outputs the correct class ID and class name. It reports failure if the outputs are not  correct. You can use the following command to run it and verify :
```
python test_onnx.py
```

##### `4. test_server.py`

This Python script tests the deployment of the model on the banana dev server. It  accepts the url of an image and returns the name of the class id the image belongs to. It also accepts a flag to run preset custom tests, where it makes calls to the deployed model using the two images from the  repository and verifies the results, similar to `test_onnx.py`. The script also reports the time it takes to make a call to the model on the server.

Run the script and provide the following arguments:

- `--image_path`: URL to the image file.
- `--test`: (optional) Run preset tests.

 You can use the following command, for example,  to run it and verify :

```
python test_server.py --test "https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01667114_mud_turtle.JPEG
```

