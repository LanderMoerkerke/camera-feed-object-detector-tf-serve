# Camera feed object detector using Tensorflow Serving

Program to detect objects using Tensorflow Detection API and YOLO on a video stream. The scripts are written in Python3.

## Dependencies

This build is based on Tensorflow Object Detection API which depends on the following libraries:

*   Protobuf 3.0.0
*   Python-tk
*   Pillow
*   Jupyter notebook
*   Matplotlib
*   Cython
*   Contextlib2
*   Tensorflow (>=1.9.0)
*   OpenCV
*   Pipenv

Luckily all these Python packages are all declared inside a Pipenv.

## Installation

### General

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

All the Python packages and Python itself should now be installed inside an virtual environment.


## Usage

Firstly, enter the virtual environment:

```bash
pipenv shell
```

### Tensorflow Serving

Install Docker.



### General

## TensorFlow Detection API

Convert model from [ zoo ](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to a TensorFlow Serving model.

First download a model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

```bash
wget "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

```bash
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models/research/object_detection
```

Follow the installation guide inside the repo.

p3 object_detection/export_inference_graph.py -input_type image_tensor --pipeline_config_path /home/lander/Documents/Howest/3NMCT/S5/Project_IV/Project/models/tf-objection-api/mask_rcnn_inception_v2_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix /home/lander/Documents/Howest/3NMCT/S5/Project_IV/Project/models/tf-objection-api/mask_rcnn_inception_v2_coco_2018_01_28/model.ckpt --output_directory ./detector2

## Run TF serve

### Perticular model

docker run -p 8501:8501 --mount type=bind,source=$PWD/mask_rcnn_inception_v2/,target=/models/half_plus_two -e MODEL_NAME=half_plus_two -t tensorflow/serving:latest

### Multiple models

- Create config file
- Create Dockerfile with all the instructions
