import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import pathlib
import tensorflow as tf
import cv2
import argparse

# calculate distance of trh object from the camera
def distance_to_camera(knownWidth, focalLength, perWidth):
  return (knownWidth * focalLength) / perWidth

# rescaling function
def rescaleFrame(frame , scale):
   width = int(frame.shape[1] * scale)
   height = int(frame.shape[0] * scale)
   dimensions = (width,height)

   return cv2.resize(frame , dimensions , interpolation=cv2.INTER_AREA)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = r'my_model_SSD_mobilnet_V3'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = r'LabelMap\Pole_label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL =PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
  
# Initialize variables
start_time = cv2.getTickCount()
frame_count = 0
KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 11.0
focal_length = 314

vid = cv2.VideoCapture(r'test_data\20221224_204515.mp4')

while True: 
  isTrue, frame = vid.read()
  
  if not isTrue:
    break
  
  frame = rescaleFrame(frame , scale=0.35)

  image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(frame)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
  detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
  detections['num_detections'] = num_detections

# detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  
  image_with_detections = frame.copy()
  
# getting the co_ordinates of the bounding box
  ymin, xmin, ymax, xmax = detections['detection_boxes'][0][:4]
  xmin, ymin, xmax, ymax = int(xmin*frame.shape[1]), int(ymin*frame.shape[0]), int(xmax*frame.shape[1]), int(ymax*frame.shape[0])
  # here the width of the bounding box in pixels is xmax-xmin
  pixel_width = xmax-xmin
  #focal_length = ((pixel_width)*KNOWN_DISTANCE)/KNOWN_WIDTH
  inches = distance_to_camera(KNOWN_WIDTH, focal_length, pixel_width)

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.5,
        agnostic_mode=False)
  
# Calculate elapsed time and fps
  elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
  fps = frame_count / elapsed_time
  
# Display text on frame
  cv2.putText(image_with_detections, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
  cv2.putText(image_with_detections, "DISTANCE: %.2fft" % (inches / 12),
		(10,55), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 0), 1)
   
  # DISPLAYS OUTPUT IMAGE
  cv2.imshow('Detector',image_with_detections)

# if key "f" is pressed then the loop will get terminated 
  if cv2.waitKey(20) & 0xFF==ord('f') :
    break
  
  # Increment frame count
  frame_count += 1

#releasing all pointers and clearing memory
vid.release()
cv2.destroyAllWindows()