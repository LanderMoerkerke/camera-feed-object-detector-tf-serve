# Camera feed object detector using Tensorflow Serving

> Program to detect objects using Tensorflow Detection API and YOLO on a video stream. The scripts are written in Python3. For YoloV3, the model is stored locally so we don't need to utilize Tensorflow Serving.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![screenshot working example](./img/header.jpg "Example")

## Table of Contents

-   [Folder structure](#folder-structure)
-   [Dependencies](#dependencies)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Tensorflow Serving](#tensorflow-serving)
    -   [Detecting Objects](#detecting-objects)
        -   [Tensorflow Object Detection API](#tensorflow-object-detection-api)
        -   [YoloV3](#yolov3)
-   [Contributing](#contributing)

## Folder structure

    .
    ├── object_detection                    # The actual code to detect objects
    │   ├── core                            # Helper functions from Tensorflow Detection API
    │   ├── data                            # Extra content (now Pickle of coco categories)
    │   ├── utils                           # Helper functions from Tensorflow Detection API
    │   └── feed.py                         # Gets a video feed, predicts ands shows the detections
    │
    ├── tf_serve                            # Dockerfile and models for Tensorflow Serving
    │   ├── config                          # Configs
    |   |   └── model_config                # Config file for the specific models
    │   ├── models                          # Neural networks, exported as Tensorflow Serving models
    │   └── Dockerfile                      # Custom build of the Tensorflow/Serving image
    │
    ├── yolo                                # YOLO object detection, without Tensorflow Serving
    │   ├── model                           # Modelfiles
    |   |   ├── coco.names                  # Coco category names
    |   |   └── darknet_yolov3_tiny.pb      # Converted model from YoloV3
    │   ├── utils                           # Helper functions from Paweł Kapica
    |   |   └── utils.py                    # Functions to create detection boxes
    │   └── feed.py                         # Gets a video feed, predicts ands shows the detections
    │
    ├── Pipfile                             # Defenition of our Python environment
    └── config.ini                          # Global defenitions of the used parameters

## Dependencies

This build is based on Tensorflow Object Detection API which depends on the following libraries:

-   Python
    -   Pillow
    -   Jupyter notebook
    -   Matplotlib
    -   Tensorflow (>=1.9.0)
    -   Requests
    -   OpenCV
    -   Pipenv
-   Docker

Luckily all these Python packages are all declared inside a Pipenv.

## Installation

Clone this repo:

```bash
git clone https://github.com/MoerkerkeLander/camera-feed-object-detector-tf-serve.git
cd camera-feed-object-detector-tf-serve
```

To enable the Python environment we use Pipenv. If you don't have this installed, we can use pip:

```bash
pip install pipenv
```

To setup Pipenv and install all the dependencies:

```bash
pipenv install -d
```

All the Python packages and the correct version of Python itself (3.6) should now be installed inside an virtual environment.

## Usage

### Config file

To share parameters between our different scripts I created a config.ini file.
Here you can configure which video feed or video file the script will be analysing and which model it will use.

**Note**: I created my own config file which is linked to my setup of cameras.
To link your cameras I propose using a feed that uses the [RTSP protocol](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol) or you can use a video file.
The keys are already predefined, so just fill in the corresponding value appropriately inside the config_template.ini.

### Tensorflow Serving

Install [Docker](https://www.docker.com/products/docker-desktop).

Build the Docker image using the configurations inside the Dockerfile. The name of the image is object-detect.

```bash
docker build -t object-detect ./tf-serve
```

Run the Docker image.

```bash
docker run --name object-detect -h 0.0.0.0 --network="host" --rm -d object-detect:latest
# --name        For easy referencing this container
# --network     Setup network as host
# --rm          Removes container when it is stopped
# -d            Daemonize the docker container
# -h            Setup hostname, so we can access it using localhost
```

Now Tensorflow Serving is running inside a Docker container. We can access it by sending a REST request or a gRPC call. I used REST in favor of gRPC because it is the simplest to setup. Inside the models directory there are three exported models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These are converted using the [export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) from the Tensorflow Object Detection API.

To check if the container is running properly:

```bash
docker ps

# OUTPUT:
> CONTAINER ID        IMAGE                  COMMAND                  CREATED             STATUS              PORTS               NAMES
> 7507a1d4e430        object-detect:latest   "tensorflow_model_se…"   15 seconds ago      Up 14 seconds                           object-detect
```

Now that Tensorflow Serving is working correctly we can start detecting some objects! You can use Tensorflow Detection API with Tensorflow Serving or YoloV3.

### Detecting Objects

Firstly, enter the virtual environment inside the root of the repository:

```bash
pipenv shell
```

Now we can execute the different Python scripts.

#### Tensorflow Object Detection API

```bash
python object_detection/feed.py
```

#### YoloV3

```bash
python yolo/feed.py
```

## Contributing

1.  Fork it (<https://github.com/MoerkerkeLander/camera-feed-object-detector-tf-serve>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a new Pull Request

## Sources

### COCO

-   [COCO Site - COCO](http://cocodataset.org/)
-   [COCO Transfer Learning - Medium](https://medium.com/practical-deep-learning/a-complete-transfer-learning-toolchain-for-semantic-segmentation-3892d722b604)

### Tensorflow Serving

-   [TF Extended - Tensorflow](https://www.tensorflow.org/tfx/)
-   [TF Serve - Medium](https://medium.com/epigramai/tensorflow-serving-101-pt-1-a79726f7c103)
-   [TF Serve - Tensorflow](https://www.tensorflow.org/serving/)
-   [TF Serve / Vitaly Bezgachev - Medium](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198)
-   [How to use TensorFlow Serving docker / Manoveg Saxena - Medium](https://medium.com/analytics-vidhya/how-to-use-tensorflow-serving-docker-container-for-model-testing-and-deployment-80a5e66322a5)
-   [TF Object Detection API / Karol Majek - Medium](https://medium.com/@karol_majek/10-simple-steps-to-tensorflow-object-detection-api-aa2e9b96dc94)
-   [TensorFlow Serving REST vs gRPC / Avidan Eran - Medium](https://medium.com/@avidaneran/tensorflow-serving-rest-vs-grpc-e8cef9d4ff62)

### Yolo

-   [Darknet Yolo - Pjreddie](https://pjreddie.com/darknet/yolo/)
-   [Tensorflow Yolo v3 - GitHub](https://github.com/mystic123/tensorflow-yolo-v3)

### Tensorflow Object Detection API

-   [Tensorflow Object Detection API - GitHub](https://github.com/tensorflow/models/tree/master/research/object_detection)
-   [TF detection model zoo - GitHub](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
-   [TensorFlow Object Detection API tutorial / Daniel Stan - Medium](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)
-   [Deploy TensorFlow object detection model / Pierre Paci - Medium](https://medium.com/@pierrepaci/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04)

<!-- ## Other less relevant sources

### Deep learning

-   [Awesome Deep Learning Papers - GitHub](https://github.com/terryum/awesome-deep-learning-papers)

### Tensorflow Extended

-   [TF Transform - Tensorflow](https://www.tensorflow.org/tfx/transform/)
-   [TF Model Analysis - Tensorflow](https://www.tensorflow.org/tfx/model_analysis/)
-   [TF Data Validation - Tensorflow](https://www.tensorflow.org/tfx/data_validation/)
-   [TF Serving Example: Tendies - GitHub](https://github.com/tmlabonte/tendies)
-   [TF Serving Example: MNIST - GitHub](https://github.com/gauravkaila/tensorflow-serving-docker-image)

### Train a object detection model

-   [LabelImg - GitHub](https://github.com/tzutalin/labelImg)
-   [Train Object Detection TensorFlow / EdjeElectronics - GitHub](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

### SSH tunnel

-   [SSH Tunnel - SSH](https://www.ssh.com/ssh/tunneling/example)) -->
