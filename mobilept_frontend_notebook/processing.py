# General imports
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime, date
import time

# Request handling and data packaging
from django.core.serializers.json import DjangoJSONEncoder
import json
import requests

# Pose tracker setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# Run BP on image
def process_image(image_path, output_dir = None):
    try:

        name = os.path.basename(os.path.normpath(image_path))
        
        # Load image.
        input_frame = cv2.imread(image_path)

        # For convrting BgR image to RGB before processing
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # Initialize fresh pose tracker and run it.
        with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
        ) as pose_tracker:

            result = pose_tracker.process(image=input_frame)

            # Get both pose in depth and real world coordinates
            # z coord tracks depth of key point
            pose_landmarks = result.pose_landmarks
            # Real world has normal x, y, z coords
            real_world_landmarks = result.pose_world_landmarks

        # Save image with pose prediction (if pose was detected).
        if output_dir is not None:
            save_image_output(output_dir, name, input_frame.copy(),pose_landmarks)
        
        landmark_output = {}
        
        # Save landmarks.
        if pose_landmarks is not None:
            # Check the number of landmarks and take pose landmarks.
            assert (
                len(pose_landmarks.landmark) == 33
            ), "Unexpected number of predicted pose landmarks: {}".format(
                len(pose_landmarks.landmark)
            )

            # Map pose landmarks from [0, 1] range to absolute coordinates to get
            # correct aspect ratio for pose_landmarks.
            frame_height, frame_width = input_frame.shape[:2]
            ratio = frame_height / frame_width
            landmarks = [
                [lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark
            ]

            # Adjustment not needed for world landmarks
            real_landmarks = [
                [lmk.x, lmk.y, lmk.z] for lmk in real_world_landmarks.landmark
            ]

            visibility = [lmk.visibility for lmk in real_world_landmarks.landmark]

            # load data to output variable
            landmark_output = {
                "pose_landmarks": landmarks,
                "pose_world_landmarks": real_landmarks,
                "visibility": visibility,
                "frame_height": frame_height,
                "frame_width": frame_width,
            }
        
        
        return True, landmark_output

    except Exception as ex:
        print(ex, file=sys.stderr)
        return False, None
    

# Save image with output
def save_image_output(path, name, frame, pose_landmarks):
    output_frame = frame.copy()
    if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path, name), output_frame)
    return output_frame


def process_video(path):
    try: 
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        output = {
            'pose_landmarks': [],
            'pose_world_landmarks': [],
            'visibilities': [],
            'time': [],
            'frame_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'frame_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        }

        frame_idx = 0

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2) as pose:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
            

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(image)


                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if result.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    

                    output['pose_landmarks'].append([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                    output['pose_world_landmarks'].append([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_world_landmarks.landmark])
                    output['visibilities'].append([lmk.visibility for lmk in result.pose_world_landmarks.landmark])
                    output['time'].append(frame_idx/fps)
                
                cv2.imshow('MediaPipe Pose', image)

                k = cv2.waitKey(5)

                frame_idx += 1

                if k == ord('q'):
                    break
            
        cap.release()

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        return True, output

    except Exception as ex:
        print(ex, file=sys.stderr)
        return False, None