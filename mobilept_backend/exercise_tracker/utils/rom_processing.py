import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import csv
import numpy as np
import os
import sys
import tqdm
from scipy.signal import savgol_filter, find_peaks

from dtw import *

from .rep_counter import FullBodyPoseEmbedder

from exercise_tracker.models import ALL_JOINTS, EXERCISE_JOINTS


# Blazpose enums for landmark data
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

JOINT_VECS = {
    'R_KNEE': [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE],
    'L_KNEE': [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE],
    'R_HIP': [RIGHT_SHOULDER,RIGHT_HIP,RIGHT_KNEE],
    'L_HIP': [LEFT_SHOULDER,LEFT_HIP,LEFT_KNEE],
    'R_ELBOW': [RIGHT_SHOULDER,RIGHT_ELBOW,RIGHT_WRIST],
    'L_ELBOW': [LEFT_SHOULDER,LEFT_ELBOW,LEFT_WRIST],
    'R_SHOULDER': [RIGHT_HIP,RIGHT_SHOULDER,RIGHT_ELBOW],
    'L_SHOULDER': [LEFT_HIP,LEFT_SHOULDER,LEFT_ELBOW],
}

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))



# Get joint angles for blazepose world landmark
def calc_joint_angles(data, signals):

    for joint, signal in signals.items():
        locs = JOINT_VECS[joint]
        first = np.array(data[locs[1]]) - np.array(data[locs[0]]) 
        second =  np.array(data[locs[1]]) - np.array(data[locs[2]]) 
        signal.append(angle_between(first,second))


def get_joint_signals(pose_kps, joints_of_int):
    signals = {joint: [] for joint in joints_of_int}

    for frame in pose_kps:
        bp_landmarks = [
            [-lmk[2], lmk[0], -lmk[1]] for lmk in frame
        ]

        calc_joint_angles(bp_landmarks, signals)
    
    return signals

def process_joint_signals(pose_kps, joints_of_int):
    try:
        joints_data = {joint: {} for joint in joints_of_int}
        
        # Get raw signals: dict map between joint code names and list of data points
        raw_signals = get_joint_signals(pose_kps, joints_of_int)

        # Smooth signals using sgolay and rolling average filter
        for joint, raw_sig in raw_signals.items():
            bp_raw = pd.Series(raw_sig)

            joints_data[joint]['raw'] = raw_sig
            joints_data[joint]['rolling'] = bp_raw.rolling(window=5, min_periods=1).mean().values.tolist()
            joints_data[joint]['s_golay'] = savgol_filter(bp_raw,10,3).tolist()

            # Convert rolling mean data to numpy array
            rolling_arr = np.array(joints_data[joint]['rolling'])

            # Calculate troughs of 'rolling' signal
            min_height = np.max(rolling_arr)*0.9
            peaks, _ = find_peaks(rolling_arr, height=min_height, distance=5)

            if len(peaks) == 0:
                joints_data[joint]['avg_max'] = 0
                joints_data[joint]['std_max'] = 0

                joints_data[joint]['peaks'] = []
                joints_data[joint]['peaks_ind'] = []
            else:
                joints_data[joint]['avg_max'] = float(rolling_arr[peaks].mean())
                joints_data[joint]['std_max'] = float(rolling_arr[peaks].std())

                joints_data[joint]['peaks'] = rolling_arr[peaks].tolist()
                joints_data[joint]['peaks_ind'] = peaks.tolist()


            # Calculate troughs of 'rolling signal
            min_height = np.max(-1*rolling_arr)*1.2
            troughs, _ = find_peaks(-1*rolling_arr, height=min_height, distance=5)

            if len(troughs) == 0:
                joints_data[joint]['avg_min'] = 0
                joints_data[joint]['std_min'] = 0

                joints_data[joint]['troughs'] = []
                joints_data[joint]['troughs_ind'] = []
            else:
                joints_data[joint]['avg_min'] = float(rolling_arr[troughs].mean())
                joints_data[joint]['std_min'] = float(rolling_arr[troughs].std())

                joints_data[joint]['troughs'] = rolling_arr[troughs].tolist()
                joints_data[joint]['troughs_ind'] = troughs.tolist()

    except Exception as ex:
            print(ex, file=sys.stderr)
            return False, None

    return True, joints_data


def dtw_exercise(exer_payload, ref_obj):
    ex_data = {
        'keypoints_sequence': exer_payload['keypoints_sequence'],
        'pose_keypoints': exer_payload['pose_keypoints'],
        'ROM': exer_payload['analytics']['ROM'],
        'exercise': exer_payload['exercise']
        }
    
    ref_data = {
        'keypoints_sequence': ref_obj.keypoints_sequence,
        'pose_keypoints': ref_obj.pose_keypoints,
        'ROM': ref_obj.analytics['ROM'],
        'exercise': ref_obj.exercise
        }
    
    flag, data = dtw_general(ex_data, ref_data)
    return flag, data

def dtw_objects(first_obj, second_obj):
    

    ex_data = {
        'keypoints_sequence': first_obj.keypoints_sequence,
        'pose_keypoints': first_obj.pose_keypoints,
        'ROM': first_obj.analytics['ROM'],
        'exercise': first_obj.exercise
    }
    
    ref_data = {
        'keypoints_sequence': second_obj.keypoints_sequence,
        'pose_keypoints': second_obj.pose_keypoints,
        'ROM': second_obj.analytics['ROM'],
        'exercise': second_obj.exercise
    }

    flag, data = dtw_general(ex_data, ref_data)
    return flag, data

def dtw_general(ex_data, ref_data):
    try:
        output = {}

        # Execute DTW with Joint Angles w/o 0 mean normalization
        # get ROM vectors with time indices placed along rows
        output['joint_angles_agg'] = dtw_joint_angles_agg(ex_data, ref_data, normalize=False, use_joints_of_int=False)
        output['joint_angles_agg_norm'] = dtw_joint_angles_agg(ex_data, ref_data, normalize=True, use_joints_of_int=False)
        output['joint_angles_agg_joi'] = dtw_joint_angles_agg(ex_data, ref_data, normalize=False, use_joints_of_int=True)
        output['joint_angles_agg_norm_joi'] = dtw_joint_angles_agg(ex_data, ref_data, normalize=True, use_joints_of_int=True)
        output['joint_angles_indiv'] = dtw_joint_angles_indiv(ex_data, ref_data, normalize=False, use_joints_of_int=False)
        output['joint_angles_indiv_norm'] = dtw_joint_angles_indiv(ex_data, ref_data, normalize=True, use_joints_of_int=False)
        output['joint_angles_indiv_joi'] = dtw_joint_angles_indiv(ex_data, ref_data, normalize=False, use_joints_of_int=True)
        output['joint_angles_indiv_norm_joi'] = dtw_joint_angles_indiv(ex_data, ref_data, normalize=True, use_joints_of_int=True)
        
        output['pose_embedder_real'] = dtw_pose_embedder(ex_data, ref_data, norm=False, real_lmks=True)
        output['pose_embedder_real_norm'] = dtw_pose_embedder(ex_data, ref_data, norm=True, real_lmks=True)
        output['pose_embedder_nonreal'] = dtw_pose_embedder(ex_data, ref_data, norm=False, real_lmks=False)
        output['pose_embedder_nonreal_norm'] = dtw_pose_embedder(ex_data, ref_data, norm=True, real_lmks=False)


        return True, output 

    except Exception as ex:
            print(ex, file=sys.stderr)
            return False, None

def dtw_joint_angles_agg(ex_data, ref_data, normalize = False, use_joints_of_int = False):
    ex_matrix = get_joint_angle_matrix(ex_data, normalize, use_joints_of_int)
    ref_matrix = get_joint_angle_matrix(ref_data, normalize, use_joints_of_int)

    alignment = dtw(ex_matrix, ref_matrix, keep_internals=True)

    num_joint_dims = ex_matrix.shape[1]

    steps = alignment.stepsTaken.shape[0]

    accuracy = 1 - (alignment.distance/(num_joint_dims*20*steps))

    return {
        'dist': alignment.distance,
        'normalized_dist': alignment.normalizedDistance,
        'path_len': steps,
        'accuracy': accuracy
    }

def get_joint_angle_matrix(ex_data, normalize=False, use_joints_of_int = False):
    joint_angle_data = []

    joints = ALL_JOINTS

    if use_joints_of_int:
        joints = EXERCISE_JOINTS[ex_data['exercise']]

    for joint in joints:
        row = np.array(ex_data['ROM'][joint]['rolling'])

        # mean normalize
        if normalize:
            row = row - row.mean()


        joint_angle_data.append(row)
    
    # Need Transpose to get time indices as rows
    return np.array(joint_angle_data).T


def dtw_joint_angles_indiv(ex_data, ref_data, normalize = False, use_joints_of_int = False):
    ex_ROM_dict = ex_data['ROM']
    ref_ROM_dict = ref_data['ROM']

    joints = ALL_JOINTS

    if use_joints_of_int:
        joints = EXERCISE_JOINTS[ex_data['exercise']]
    
    dtw_indiv_results = {}

    norm_sum = 0
    dist_sum = 0

    for joint in joints:

        ref_roll = np.array(ref_ROM_dict[joint]['rolling'])
        ex_roll = np.array(ex_ROM_dict[joint]['rolling'])

        if normalize:
            ref_roll = ref_roll - ref_roll.mean()
            ex_roll = ex_roll - ex_roll.mean()

        # calculate individual DTW distance
        alignment = dtw(ex_roll, ref_roll, keep_internals=True)
        steps = alignment.stepsTaken.shape[0]

        dtw_indiv_results[joint] = {
            'dist': alignment.distance,
            'normalized_dist': alignment.normalizedDistance,
            'path_len': steps,
        }

        norm_sum += alignment.normalizedDistance
        dist_sum += alignment.distance

    accuracy = 1 - (norm_sum/(len(joints)*20))

    return {
        'joint_alignments': dtw_indiv_results,
        'dist': dist_sum,
        'normalized_dist': norm_sum,
        'accuracy': accuracy
    }

def dtw_pose_embedder(ex_data, ref_data, norm = False, real_lmks = True):

    pose_embedder = FullBodyPoseEmbedder()

    # Choose to use either pose_lmks or pose_world_lmks
    if real_lmks:
        ex_matrix = get_pose_embedder_matrix(ex_data['keypoints_sequence']['pose_world_landmarks'], pose_embedder, norm)
        ref_matrix = get_pose_embedder_matrix(ref_data['keypoints_sequence']['pose_world_landmarks'], pose_embedder, norm)
    else:
        ex_matrix = get_pose_embedder_matrix(ex_data['keypoints_sequence']['pose_landmarks'], pose_embedder, norm)
        ref_matrix = get_pose_embedder_matrix(ref_data['keypoints_sequence']['pose_landmarks'], pose_embedder, norm)

    alignment = dtw(ex_matrix, ref_matrix, keep_internals=True)

    num_joint_dims = ex_matrix.shape[1]

    steps = alignment.stepsTaken.shape[0]

    accuracy = 1 - (alignment.distance/(num_joint_dims*1*steps))


    return {
        'dist': alignment.distance,
        'normalized_dist': alignment.normalizedDistance,
        'path_len': steps,
        'accuracy': accuracy
    }

    
def get_pose_embedder_matrix(pose_lmks, pose_embedder, norm=False):
    
    embeddings =  [pose_embedder(np.array(frame)) for frame in pose_lmks]
    if norm:
        return np.array([np.linalg.norm(embedding, axis=1).flatten() for embedding in embeddings])
    else:
        return np.array([embedding.flatten() for embedding in embeddings])


