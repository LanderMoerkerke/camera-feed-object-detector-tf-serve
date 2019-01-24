import os
import configparser
import cv2
import numpy as np
import pickle
import queue
import requests
import sys
import threading
import time
import datetime
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


def save_detection(classname, image):
    """Saves a image
    :classname: name of the class
    :image: image to save
    """

    date = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(f"{_SAVE_DIR}/{classname}", exist_ok=True)

    return cv2.imwrite(f"{_SAVE_DIR}/{classname}/{date}-{classname}.jpg", frame)


# ------------------
# CONFIGURATION
# ------------------

_CONFIG_FILE = "config.ini"

config = configparser.ConfigParser()
config.read(_CONFIG_FILE)

_TF_SERVING_URL = config["Tensorflow"]["tf_serving_url"]
_FILE_LABELS = "coco"
_THRESHOLD = 0.5

_SAVE_DETECTION = config["Tensorflow"].getboolean("save_detection")
_SAVE_DIR = config["Tensorflow"]["save_dir"]


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
# QUEUE & THREADING RETRIEVING
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
        target=retrieve_frames,
        name=f"thread-retrieve-{i}",
        kwargs={"cap": cap},
        daemon=True
    )

    th_retrieve_frames.start()
    time.sleep(1)


# ------------------
# QUEUE & THREADING DETECTIONS
# ------------------

detection_queue = queue.Queue()


def handle_detections():
    vehicles = [
        "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "skateboard"
    ]

    animals = [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe"
    ]

    while True:
        frame, detections = detection_queue.get()
        print(f"Got detections at {time.time()}:")
        pprint(detections)

        for detection in detections:
            if detection["class"] in "person":
                if _SAVE_DETECTION:
                    save_detection(detection["class"], frame)
            elif detection["class"] in vehicles:
                if _SAVE_DETECTION:
                    save_detection(detection["class"], frame)
            elif detection["class"] in animals:
                if _SAVE_DETECTION:
                    save_detection(detection["class"], frame)
        detection_queue.task_done()


th_detections = threading.Thread(
    target=handle_detections,
    name=f"thread-detect-{i}",
    daemon=True
)

th_detections.start()

# -------------------
# PROCESSING
# -------------------

print("Starting detection")
while(True):

    # frame video file
    print("New frame queue item. Amount of frames in queue:",
          frame_queue.qsize())

    frame, img_processed, payload = frame_queue.get()
    frame_queue.task_done()
    # print(img_processed.shape)

    # detecting objects
    # t0 = time.time()
    # payload = {"instances": img_processed.tolist()}
    # print("Amount of seconds to create payload:", time.time() - t0)

    t0 = time.time()
    try:
        res = requests.post(
            _TF_SERVING_URL,
            json=payload
        )
    except requests.exceptions.RequestException:
        print("ERROR: Request error, did you start Tensorflow Serving?")
        sys.exit()
    except Exception as e:
        raise e
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

    detection_queue.put((img_processed[0], detections))

    # close windows when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        retrieving_frames = False
        cv2.destroyAllWindows()

        # exit script
        print("Exiting script")
        sys.exit(1)

        break
