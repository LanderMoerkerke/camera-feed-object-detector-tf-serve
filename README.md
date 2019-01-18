# Camera feed object detector using Tensorflow Serving
> Program to detect objects using Tensorflow Detection API and YOLO on a video stream. The scripts are written in Python3. For YoloV3, the model is stored locally so we don't need to utilize Tensorflow Serving.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<!-- vim-markdown-toc GitLab -->

* [Folder structure](#folder-structure)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)
    * [Tensorflow Serving](#tensorflow-serving)
    * [Detecting Objects](#detecting-objects)
        * [Tensorflow Object Detection API](#tensorflow-object-detection-api)
        * [YoloV3](#yolov3)
* [Contributing](#contributing)

<!-- vim-markdown-toc -->


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
    │   ├── utils                           # Helper functions from TODO
    |   |   └── utils.py                    # Functions to create detection boxes
    │   └── feed.py                         # Gets a video feed, predicts ands shows the detections
    │
    ├── Pipfile                             # Defenition of our Python environment
    └── config                              # Defenitions of the used parameters

## Dependencies

This build is based on Tensorflow Object Detection API which depends on the following libraries:

- Python
    *   Pillow
    *   Jupyter notebook
    *   Matplotlib
    *   Tensorflow (>=1.9.0)
    *   Requests
    *   OpenCV
    *   Pipenv
- Docker

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

### Tensorflow Serving

Install [Docker](https://www.docker.com/products/docker-desktop).

Build the Docker image using the configurations inside the Dockerfile. The name of the image is object-detect.

```bash
cd tf-serve
docker build -t object-detect .
```

Run the Docker image.

```bash
docker run --name object-detect -h 0.0.0.0 --network="host" --rm object-detect:latest
# --name        For easy referencing this container
# --network     Setup network as host
# --rm          Removes container when it is stopped
# -h            Setup hostname, so we can access it using localhost
```

Now Tensorflow Serving is running inside a Docker container. We can access it by sending a REST request or a gRPC call. We used REST in favor of gRPC because it is the simplest to setup. Inside the models directory there are three exported models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These are converted using the [export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) from the Tensorflow Object Detection API.

To check if the container is running properly:

```bash
docker ps

# OUTPUT:
> CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS               NAMES
> fb18e78e71aa        object-detect       "tensorflow_model_se…"   11 seconds ago      Up 10 seconds                           object-detect
```

Now that Tensorflow Serving is working correctly we can start detecting some objects! We can use Tensorflow Detection API with Tensorflow Serving or YoloV3.

### Detecting Objects

Firstly, enter the virtual environment inside the root of the repository:

```bash
pipenv shell
```

Now we can execute the different Python scripts.

**Note**: these scripts are linked to my setup of cameras. To link your cameras we propose using a feed that uses the [RTSP protocol](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol) or you can use a video file. We defined these parameters previously inside the config file.

#### Tensorflow Object Detection API

```bash
python object_detection/feed.py
```

#### YoloV3

```bash
python yolo/feed.py
```

## Contributing

1. Fork it (<https://github.com/MoerkerkeLander/camera-feed-object-detector-tf-serve>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
