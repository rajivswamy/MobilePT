import numpy
import numpy as np
from dtw import *
import mediapipe as mp
import matplotlib.pyplot as plt

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from matplotlib import animation
from matplotlib.animation import FuncAnimation
from plotly.subplots import make_subplots

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.3


EXERCISE_JOINTS = {'SQUAT': ['L_KNEE', 'R_KNEE', 'L_HIP','R_HIP'],
                     'ARM_RAISE': ['L_SHOULDER','R_SHOULDER'],
                     'LEG_RAISE': ['L_HIP','R_HIP'],
                     'BICEP_CURL':['L_ELBOW','R_ELBOW']}

ALL_JOINTS = ['L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP', 'L_ELBOW', 'R_ELBOW', 
              'L_SHOULDER', 'R_SHOULDER']





def plot_ROM_dtw_comparisons(ex_data, ref_data):
    ex_ROM_dict = ex_data['analytics']['ROM']
    ref_ROM_dict = ref_data['analytics']['ROM']

    for joint in ALL_JOINTS:
        fig, axs = plt.subplots(ncols = 1, nrows=2, figsize=(20, 10), sharey=True)

        # plot reference
        ref = axs[0]

        # get data
        ref_roll = np.array(ref_ROM_dict[joint]['rolling'])
        x = range(len(ref_roll))
        ref.plot(x, ref_roll, label='Rolling Average')
        ref.set(xlabel='Frame', ylabel='Joint Angle (degrees)', title=f'{joint}: ROM Reference')



        # plot exercise
        ex = axs[1]
        ex_roll = np.array(ex_ROM_dict[joint]['rolling'])
        x = range(len(ex_roll))
        ex.plot(x, ex_roll, label='Rolling Average')
        ex.set(xlabel='Frame', ylabel='Joint Angle (degrees)', title=f'{joint}: ROM Exercise Session')


        # calculate individual DTW distance
        alignment = dtw(ex_roll, ref_roll, keep_internals=True)
        alignment.plot('twoway', offset=100, xlab='Frame', ylab='Joint Angle (degrees)', label=f"DTW Alignment: {joint}")

def plot_joint_angles(data):
    rom_data = data['analytics']['ROM']

    for joint, data in rom_data.items():
        fig, ax = plt.subplots()
        roll = data['rolling']
        x = range(len(roll))

        ax.plot(x,roll, label='Rolling Average')
        ax.plot(data['peaks_ind'], data['peaks'], "x", label='Peaks', color='green')
        ax.plot(data['troughs_ind'], data['troughs'], "x", label='Troughs', color='red')
        plt.axhline(y=data['avg_max'], color='gray', linestyle='-.', label='Average Max ROM')
        plt.axhline(y=data['avg_min'], color='firebrick', linestyle='-.', label = 'Average Min ROM')


        ax.set_title(f'{joint} ROM Data', fontsize=14)
        ax.set_xlabel('Frame', fontsize=14)
        ax.set_ylabel('Joint Angle (degrees)', fontsize=14)

        ax.legend()

def calc_DTW_joints_metrics(ex_data, ref_data):

    ex_ROM_dict = ex_data['analytics']['ROM']
    ref_ROM_dict = ref_data['analytics']['ROM']

    dtw_results = {}

    norm_sum = 0
    dist_sum = 0

    for joint in ALL_JOINTS:

        ref_roll = np.array(ref_ROM_dict[joint]['rolling'])
    
        ex_roll = np.array(ex_ROM_dict[joint]['rolling'])


        # calculate individual DTW distance
        alignment = dtw(ex_roll, ref_roll, keep_internals=True)
        steps = alignment.stepsTaken.shape[0]

        dtw_results[joint] = {
            'total': alignment.distance,
            'norm': alignment.normalizedDistance,
            'path_len': steps,
        }

        norm_sum += alignment.normalizedDistance
        dist_sum += alignment.distance


    dtw_results['aggregate'] = ex_data['analytics']['DTW']['joint_angles_agg']
    dtw_results['norm_sum'] = norm_sum
    dtw_results['dist_sum'] = dist_sum


    return dtw_results


# Adapted from: https://stackoverflow.com/questions/69265059/is-it-possible-to-create-a-plotly-animated-3d-scatter-plot-of-mediapipes-body-p
def plot_landmarks_custom(
    landmark_list, visibility,
    connections=None,
):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list):
        if (
            visibility[idx] < _VISIBILITY_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark[2], landmark[0], -landmark[1])
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list)

        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [   
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig


def plot_ex_animation(ex_data):
    animate_lmks(ex_data['keypoints_sequence']['pose_world_landmarks'], ex_data['keypoints_sequence']['visibilities'])

def animate_lmks(world_kps, vis):
    zip_data = zip(world_kps, vis)


    frames = []

    for frame in zip_data:
        world_kps = frame[0]
        vis = frame[1]

        df, cn2 = animate_helper(world_kps, vis, mp_pose.POSE_CONNECTIONS)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=df["z"], y=df["x"], z=df["y"], mode='markers'),
                    go.Scatter3d(
                        x=cn2["xs"],
                        y=cn2["ys"],
                        z=cn2["zs"],
                        mode="lines",
                        line={"color": "black", "width": 5},
                        name="connections",
                    )
                ]
            )
        )
    
    fig = go.Figure(
        data=[
            go.Scatter3d() for traces in frames[0]
        ],
        layout=go.Layout( # Styling
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None],
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )
    fig.show()


    

def animate_helper(landmark_list, visibility, connections=None):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list):
        if (
            visibility[idx] < _VISIBILITY_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark[2], landmark[0], -landmark[1])

    
    cn2 = None 

    if connections:
        out_cn = []
        num_landmarks = len(landmark_list)

        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values

    return df, cn2


# Adapted from Infinity AI
def create_3D_keypoints_video(
    output_path: str,
    fps: int,
    ex_data,
    dpi: int = 150,
    width_in_pixels: int = 200,
    height_in_pixels: int = 200,
) -> str:
    """Visualizes 3D keypoint annotations as video.
    Args:
        output_path: Path to output video
        fps: desired frame rate of output video
        coco: COCO data
        dpi: resolution (dots per inch)
        width_in_pixels: Width of figure, in pixels
        height_in_pixels: Height of figure, in pixels
    Returns:
        Path to resulting animation video.
    """
    fig_height = height_in_pixels / dpi
    fig_width = width_in_pixels / dpi
    figsize = (fig_width, fig_height)
    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    global_coords = []

    world_kps = ex_data['keypoints_sequence']['pose_world_landmarks']
    # vis = ex_data['keypoints_sequence']['visibilities']

    # zip_data = zip(world_kps, vis)

    for frame in world_kps:
        global_coords.append([[-lmk[2], lmk[0], -lmk[1]] for lmk in frame])
    
    global_coords = np.array(global_coords)
    

    def update(num):
        graph._offsets3d = (
            global_coords[num, :, 0],
            global_coords[num, :, 1],
            global_coords[num, :, 2],
        )

    graph = ax.scatter(global_coords[0, :, 0], global_coords[0, :, 1], global_coords[0, :, 2])

    ax.set_box_aspect(np.ptp(global_coords, axis=(0, 1)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid("off")

    ani = animation.FuncAnimation(fig, update, global_coords.shape[0], blit=False, interval=1000 / fps)
    ani.save(output_path)
    plt.close()
    return output_path