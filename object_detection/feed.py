import configparser
import cv2
import numpy as np
import pickle
import queue
import requests
import sys
import threading
import time
from pprint import pprint
from utils import output_utlis as out_util
from utils import visualization_utils as vis_util

# -------------------
# FUNCTIONS
# -------------------


def save_obj(obj, name):
    with open(
        'object_detection/data/' + name + '.pkl',
        'wb'
    ) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('object_detection/data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def expand_image(img):
    return np.expand_dims(img, axis=0)


def preprocess_frame(frame):
    preprocessed_img = expand_image(frame)
    payload = {"instances": preprocessed_img.tolist()}
    return frame, preprocessed_img, payload


# ------------------
# CONFIGURATION
# ------------------

_CONFIG_FILE = "config.ini"

config = configparser.ConfigParser()
config.read(_CONFIG_FILE)

_TF_SERVING_URL = config["Tensorflow"]["tf_serving_url"]
_FILE_LABELS = "coco"
_THRESHOLD = 0.5
_SAVE_DETECTION = config["Tensorflow"]["save_detection"]

# ------------------
# VIDEO FEED URL
# ------------------

_DETECTION_SOURCE = config["General"]["detection_source"]


# -------------------
# INITIALISATION
# -------------------

cap = cv2.VideoCapture(
    _DETECTION_SOURCE
)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ret, frame = cap.read(1)
_HEIGHT, _WIDTH, _ = frame.shape

if not cap.isOpened():
    print("Cannot read video stream, exiting...")
    sys.exit(1)

# load labels
classes = load_obj(_FILE_LABELS)

# ------------------
# QUEUE & THREAD
# ------------------

frame_queue = queue.LifoQueue(5)

lock = threading.Lock()
frame_counter = 0


def retrieve_frames(cap):
    global frame_counter

    print("Retrieving frames")
    while retrieving_frames:
        t0 = time.time()
        ret, frame = cap.read(1)
        print("Amount of seconds to get frame:", time.time() - t0)

        print(f"[id: {frame_counter}] Got frame")
        # if frame:
        frame_queue.put(preprocess_frame(frame))
        print(f"[id: {frame_counter}] Processed frame")
        # time.sleep(0.2)

        while frame_queue.full():
            cap.grab()
            lock.acquire()
            frame_counter += 1
            lock.release()
            # time.sleep(0.005)

        lock.acquire()
        frame_counter += 1
        lock.release()


for i in range(1):
    retrieving_frames = True
    th_retrieve_frames = threading.Thread(
        target=retrieve_frames, name=f"thread-{i}", kwargs={"cap": cap}
    )

    th_retrieve_frames.start()
    time.sleep(1)

# -------------------
# PROCESSING
# -------------------

print("Starting detection")
while(True):

    # frame video file
    print("New queue item. Amount of queue items:", frame_queue.qsize())

    frame, img_processed, payload = frame_queue.get()
    frame_queue.task_done()
    # print(img_processed.shape)

    # detecting objects
    # t0 = time.time()
    # payload = {"instances": img_processed.tolist()}
    # print("Amount of seconds to create payload:", time.time() - t0)

    t0 = time.time()
    res = requests.post(
        _TF_SERVING_URL,
        json=payload
    )
    print("Amount of seconds to predict:", time.time() - t0)

    if (res.status_code == 400):
        print("Error:", res.text)
        pass
    else:
        t0 = time.time()
        output_dict = res.json()["predictions"][0]
        print("Amount of seconds to get JSON:", time.time() - t0)

        t0 = time.time()
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.array(output_dict['detection_boxes']),
            np.array(output_dict['detection_classes'], dtype="uint8"),
            output_dict['detection_scores'],
            classes,
            # instance_masks=output_dict['detection_masks'],
            use_normalized_coordinates=True,
            line_thickness=2
        )
        print("Amount of seconds to visualize:", time.time() - t0)

    # show frame to user
    t0 = time.time()
    cv2.imshow('frame', img_processed[0])
    print("Amount of seconds to show image:", time.time() - t0)

    detections = out_util.convert_output_to_detections(
        output_dict, classes, _THRESHOLD, _WIDTH, _HEIGHT)

    print(detections)

    # close windows when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        retrieving_frames = False
        cv2.destroyAllWindows()

        # exit script
        print("Exiting script")
        sys.exit(1)

        break
