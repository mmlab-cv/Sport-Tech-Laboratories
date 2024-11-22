import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):

  joints_selected = [0,11,12,13,14,15,16,23,24,25,26,27,28]
  
  edges = [(1,2),(2,4),(4,6),(1,3),(3,5),(1,7),(2,8),(8,7),(8,10),(10,12),(7,9),(9,11)]
  pose_landmarks_list = [[detection_result.pose_landmarks[0][i] for i in joints_selected]]
  # pose_landmarks_list = detection_result.pose_landmarks

  print(pose_landmarks_list)
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks_person = pose_landmarks_list[idx]
    pose_landmarks = [pose_landmarks_person[i] for i in joints_selected]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    # solutions.drawing_utils.draw_landmarks(
    #   annotated_image,
    #   pose_landmarks_proto,
    #   solutions.pose.POSE_CONNECTIONS,
    #   solutions.drawing_styles.get_default_pose_landmarks_style())

    solutions.drawing_utils.draw_landmarks(
    annotated_image,
    pose_landmarks_proto,
    edges,
    solutions.drawing_styles.get_default_pose_landmarks_style())


  return annotated_image
# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(("/teamspace/studios/this_studio/COCO_sport_samples/000000037988.jpg"))

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

cv2.imwrite('resulting_pose.png', image_rgb)