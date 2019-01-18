import cv2
import numpy as np
import pickle
import queue
import requests
import sys
import threading
import time
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
# VIDEO FEED URL
# ------------------


_URL_OPRIT = 'rtsp://Bezoeker:Test123@192.168.0.60:88/videoMain'
_URL_OPRIT_PUBLIC = 'rtsp://Bezoeker:Test123@localhost:8888/videoMain'

_URL_VOORDEUR = 'rtsp://Bezoeker:Test123@192.168.0.61:88/videoMain'
_URL_VOORDEUR_PUBLIC = 'rtsp://Bezoeker:Test123@localhost:8889/videoSub'
_PATH_VIDEO = "/mnt/ssd/Oprit/20181123AM-1-Afzagen/Oprit-20181123-083146-1542958306.mp4"


# ------------------
# SETTINGS
# ------------------

_FILE_LABELS = "coco"

# -------------------
# INITIALISATION
# -------------------

cap = cv2.VideoCapture(
    # _URL_OPRIT_PUBLIC
    # _URL_OPRIT
    _PATH_VIDEO
    # _URL_VOORDEUR_PUBLIC
)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot read video stream, exiting...")
    sys.exit(1)

# load labels
classes = load_obj(_FILE_LABELS)

# ------------------
# QUEUE & THREAD
# ------------------

frame_queue = queue.LifoQueue(1)

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

# def retrieve_frames(cap):
#     global frame_counter
#
#     print("Retrieving frames")
#     while retrieving_frames:
#         lock.acquire()
#         cap.set(1, frame_counter)
#         lock.release()
#
#         t0 = time.time()
#         ret, frame = cap.read(1)
#         print("Amount of seconds to get frame:", time.time() - t0)
#
#         print(f"[id: {frame_counter}] Got frame")
#         # if frame:
#         frame_queue.put(preprocess_frame(frame))
#         print(f"[id: {frame_counter}] Processed frame")
#         # time.sleep(0.2)
#
#         lock.acquire()
#         frame_counter += 1
#         lock.release()


for i in range(1):
    retrieving_frames = True
    th_retrieve_frames = threading.Thread(
        target=retrieve_frames, name=f"thread-{i}", kwargs={"cap": cap}
    )

    th_retrieve_frames.start()

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
        # "http://localhost:8501/v1/models/mask_rcnn_inception_v2:predict",
        # "http://172.17.0.2:8501/v1/models/faster_rcnn_inception_v2:predict",
        "http://localhost:8501/v1/models/ssd_mobilenet_v1_coco:predict",
        json=payload
    )
    print("Amount of seconds to predict:", time.time() - t0)

    if (res.status_code == 400):
        # print("Error:", res.text)
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

    # close windows when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        retrieving_frames = False
        cv2.destroyAllWindows()

        # exit script
        print("Exiting script")
        sys.exit(1)

        break
