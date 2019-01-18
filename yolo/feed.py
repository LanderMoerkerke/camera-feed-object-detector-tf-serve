import sys
import cv2
import time
import threading
import queue
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
    load_graph, letter_box_image

# -------------------
# FUNCTIONS
# -------------------


def convert_cv_to_pil(frame):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_im)
    return pil_img


def resize_pil_image(image):
    img_resized = letter_box_image(image, _INPUT_SIZE, _INPUT_SIZE, 128)
    return img_resized.astype(np.float32)


def preprocess_frame(frame):
    pil_img = convert_cv_to_pil(frame)
    resized_img = resize_pil_image(pil_img)

    return pil_img, resized_img


# ------------------
# VIDEO FEED URL
# ------------------

_URL_OPRIT = 'rtsp://Bezoeker:Test123@192.168.0.60:88/videoMain'
_URL_OPRIT_PUBLIC = 'rtsp://Bezoeker:Test123@localhost:8888/videoMain'
# _URL_OPRIT = 'rtsp://Bezoeker:Test123@localhost:8888/videoMain'
_URL_VOORDEUR = 'rtsp://Bezoeker:Test123@192.168.0.61:88/videoMain'
_URL_VOORDEUR_PUBLIC = 'rtsp://Bezoeker:Test123@localhost:8889/videoSub'


# ------------------
# SETTINGS
# ------------------

# YOLO
_INPUT_SIZE = 416
_COCO_FILE = './yolo/model/coco.names'
_FROZEN_MODEL = './yolo/model/darknet_yolov3_tiny.pb'
# _FROZEN_MODEL = './yolo/model/frozen_darknet_yolov3_model_tiny.pb'

# min acc
_CONF_THRESHOLD = 0.6
# min intersaction
_IOU_THRESHOLD = 0.5

# -------------------
# INITIALISATION
# -------------------

cap = cv2.VideoCapture(
    # _URL_OPRIT_PUBLIC
    _URL_VOORDEUR_PUBLIC
)

if not cap.isOpened():
    print("Cannot read video stream, exiting...")
    sys.exit(1)

# cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number-1)

classes = load_coco_names(_COCO_FILE)
frozen_graph = load_graph(_FROZEN_MODEL)
boxes, inputs = get_boxes_and_inputs_pb(frozen_graph)


# ------------------
# QUEUE & THREAD
# ------------------

frame_queue = queue.LifoQueue(1)


def retrieve_frames(cap):
    print("Retrieving frames")
    frame_counter = 0
    while retrieving_frames:
        cap.set(1, frame_counter)
        ret, frame = cap.read(1)
        print("Got frame")
        # img, img_processed = preprocess_frame(frame)
        frame_queue.put(preprocess_frame(frame))
        frame_counter += 1
        time.sleep(0.5)


retrieving_frames = True
th_retrieve_frames = threading.Thread(
    target=retrieve_frames, name="thread-0", kwargs={"cap": cap})

th_retrieve_frames.start()

# -------------------
# PROCESSING
# -------------------

print("Loading session")
with tf.Session(graph=frozen_graph) as sess:

    while(True):

        # frame video file
        print("Analysing new frame")

        # # cap.get(r frame_counter)
        # ret, frame = cap.read(1)
        # frame_counter += 1

        img, img_processed = frame_queue.get()

        if img is None or img_processed is None:
            break

        # detecting objects
        t0 = time.time()
        detected_boxes = sess.run(boxes, feed_dict={inputs: [img_processed]})
        t1 = time.time()

        print("Amount of seconds to predict:", t1 - t0)

        # non max supression
        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=_CONF_THRESHOLD,
                                             iou_threshold=_IOU_THRESHOLD)
        # print("Predictions found in {:.2f}s".format(time.time() - t0))

        draw_boxes(filtered_boxes, img, classes,
                   (_INPUT_SIZE, _INPUT_SIZE), True)

        open_cv_image = np.array(img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # show frame to user
        cv2.imshow('frame', open_cv_image)

        # close windows when pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            retrieving_frames = False
            cv2.destroyAllWindows()

            # exit script
            print("Exiting script")
            sys.exit(1)

            break
