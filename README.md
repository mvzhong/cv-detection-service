# cv-detection-service
Small service that runs OpenCV detection on base64 or image files. Uses a SSD Mobilenet V3 model pretrained on COCO data set.

Includes a simple RESTful API built with FastAPI that takes an image and returns a list of detected objects, their labels, the confidence level, and the percentage of the total image area the bounding boxes occupy.

I'm running this on a local server and sending images from an ESP32-CAM module for classification. I'm planning to use the labels to trigger a spray can whenever my cats get on the counter :)

Client code for ESP32-CAM streaming image to this CV service: https://github.com/mvzhong/esp32-cv-stream

Example output labeled by the server (using random image found online):
![sample good output](https://github.com/mvzhong/cv-detection-service/blob/main/readme-files/sample-output.png?raw=true)

Unfortunately, the labeling is not always the best, and will likely require me to train the models myself on more cat data, or on my cats specifically:
![sample bad output](https://github.com/mvzhong/cv-detection-service/blob/main/readme-files/skateboard.png?raw=true) 

## Setup
This is a Python3 project with [PDM](https://pdm-project.org/en/latest/) as the package manager. You'll need to install PDM to get all the dependencies installed.

You also need Python 3.11.* or greater installed.

Once you have that done, you can run `pdm install` from the root directory of this project, and it should handle installing all dependencies.

## Running
Boot up a Uvicorn server to serve the API with `pdm run scripts/start_api.py`.
