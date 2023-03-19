'''
Perform video augmentation using rotation matrices and mediapipe coordinates
- Augmentation performed only on original video and horizontally flipped video 
    -> Ensure equal augmentation for left and right handed signers
- Small angle rotations in x, y and z directions
    -> Augmentation Factor: 3 (1, 2, 5 degrees) * 2 (+/- degrees) * 3 (x, y, z rotation) * 2 (original, horizontal aug) = 36 times per video!

- Execution Time: 1m 30s for 5 signs

- New Execution Time: 1933s ~ 32m for 5 signs
- Newer Execution Time: 382s ~ 5m for 5 more signs

- Latest Execution Time: 1345 ~ 22m for 10 more signs

Reflection / Horizontal Flip (vidaug)
Rotation (x, y, z)
Dilation / Scale

Todo:
Translation
Shear
'''

import os
import math
import time
import torch
import random

PREPROCESS_PATH = "../../../inputs/interim"

MM_RESHAPE = (67, 3)
NN_RESHAPE = (-1, )

def matrix_multiplication(coordinates, rotation_matrix):
    # [x, y, z] * rotation_matrix
    return torch.mm(coordinates, rotation_matrix)

def collect_aug_rot(frames, rotation_matrix):
    aug_frames = []

    # Perform augmentation
    for i, frame in enumerate(frames):
        # Detach frame to make changes and reshape for quick matrix multiplication
        aug_frame = frame.detach().clone()
        aug_frame = torch.reshape(aug_frame, MM_RESHAPE)

        aug_frame = matrix_multiplication(aug_frame, rotation_matrix)
        aug_frames.append(torch.reshape(aug_frame, NN_RESHAPE))
    
    return aug_frames

# TODO: IP: Need to modify translation script
def perform_aug_trans(frames, pxz, py, ones_matrix):
    aug_frames = []

    # Perform augmentation
    for i, frame in enumerate(frames):
        # Detach frame to make changes and reshape for quick matrix multiplication
        aug_frame = frame.detach().clone()
        aug_frame = torch.reshape(aug_frame, MM_RESHAPE)
        aug_frame = torch.cat([aug_frame, ones_matrix], -1)

        # TODO: Get x, y, z values to determine shift amounts
        # Pick middle value for each hand and pose instead of just first value
        # ax, ay, az, _ = aug_frame[0]
        # dx = pxz * ax
        # dy = py * ay
        # dz = py * az
        # trans_matrix = get_translation_matrix(dx, dy, dz)

        # TODO: Check if 0s for hands then don't shift (not detected case)

        # TODO: Write script to get all x, y, z (individually) values for hand, hand, pose to compare 

        aug_frame = matrix_multiplication(aug_frame, translation_matrix)
        aug_frame = aug_frame[:,:-1]
        aug_frames.append(torch.reshape(aug_frame, NN_RESHAPE))
    
    return aug_frames
   

# Return rotation matrix with theta
def get_rotation_matrix_x(theta):
    return torch.tensor([ 
                [ 1, 0, 0 ],
                [ 0, math.cos(theta), -1*math.sin(theta) ],
                [ 0, math.sin(theta), math.cos(theta) ]
            ])

def get_rotation_matrix_y(theta):
    return torch.tensor([ 
                [ math.cos(theta), 0, math.sin(theta) ], 
                [ 0, 1, 0], 
                [ -1*math.sin(theta), 0, math.cos(theta)]
            ])

def get_rotation_matrix_z(theta):
    return torch.tensor([ 
                [ math.cos(theta), -1*math.sin(theta), 0 ], 
                [ math.sin(theta), math.cos(theta), 0 ], 
                [ 0, 0, 1 ]
            ])

# TODO: IP: Need to find correct values to translate by
def get_translation_matrix(dx, dy, dz):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [dx, dy, dz, 1]
    ])

def collect_aug_scale(frames, factor):
    aug_frames = []

    for frame in frames:
        # Detach frame to make changes
        aug_frame = frame.detach().clone()

        # Perform scaling by factor
        aug_frame = aug_frame * torch.tensor([factor])
        aug_frames.append(aug_frame)
    
    return aug_frames

def load_sign_video(preprocess_folder, video):
    return torch.load(f"{preprocess_folder}/{video}")

def main():
    start_time = time.time()

    # Rotation degrees, scaling factors and translation percentages
    degs = [-5, -4, 4, 5]
    # [-3, -2, -1, 1, 2, 3]
    facs = [0.95, 0.975, 0.99, 1.01, 1.025, 1.05]
    percs = [-4.5, -3, -1.5, 1.5, 3, 4.5]
    # x: -3e-02, y: +4e-02, z: 

    # Get all actions/gestures names
    actions = os.listdir(PREPROCESS_PATH)
    num_actions = len(actions)
    for i, action in enumerate(actions):
        print(f"\n--- Starting {action} matrix augmentation ({i+1}/{num_actions}) ---")
        preprocess_folder = f"{PREPROCESS_PATH}/{action}"

        videos = os.listdir(preprocess_folder)
        for video in videos:
            vid_prefix = video.split('.pt')[0]
            sign_name, video_no, *_ = vid_prefix.split('_') 

            # Skip file if already augmented 
            # TODO: '_trans_' in vid_prefix or os.path.exists(f"{PREPROCESS_PATH}/{action}/{vid_prefix}_trans_p=0.99.pt")
            if '_rot_' in vid_prefix or '_scale_' in vid_prefix:
                # or os.path.exists(f"{PREPROCESS_PATH}/{action}/{vid_prefix}_rot_x=1.pt"):
                continue
            
            # Load data instance
            frames = load_sign_video(preprocess_folder, video)

            '''
            # Perform rotation with multiple degrees
            for degree in degs:
                theta = math.radians(degree)

                # Define rotation matrix constants
                x_rotation_matrix = get_rotation_matrix_x(theta)
                y_rotation_matrix = get_rotation_matrix_y(theta)
                z_rotation_matrix = get_rotation_matrix_z(theta)

                # Augment frames using rotation matrices
                aug_x_frames = collect_aug_rot(frames, x_rotation_matrix)
                aug_y_frames = collect_aug_rot(frames, y_rotation_matrix)
                aug_z_frames = collect_aug_rot(frames, z_rotation_matrix)
      
                # Save augmented frames as torch files
                # torch.save(aug_x_frames, f'{preprocess_folder}/{vid_prefix}_rot_x={degree}.pt')
                # torch.save(aug_y_frames, f'{preprocess_folder}/{vid_prefix}_rot_y={degree}.pt')
                # torch.save(aug_z_frames, f'{preprocess_folder}/{vid_prefix}_rot_z={degree}.pt')
            '''

            '''
            # Perform dilation/scaling with multiple factors
            for factor in facs:
                # Augment frames using simple multiplicative dilation
                aug_scale_frames = collect_aug_scale(frames, factor)

                # Save augmented frames as torch files
                torch.save(aug_scale_frames, f'{preprocess_folder}/{vid_prefix}_scale_k={factor}.pt')
            '''

            # Perform translation with multiple percentages
            '''
            #TODO: Work in progress
            for x_z_perc in percs:
                for y_perc in percs:
                    # Augment frames using translation matrix
                    aug_trans_frames = collect_aug_trans(frames, x_z_perc, y_perc, ones)

                    # Save augmented frames as torch files
                    # torch.save(aug_scale_frames, f'{preprocess_folder}/{vid_prefix}_trans_x/z={x_z_perc}_y={y_perc}.pt')
            '''

            # Perform multiple rotations
            '''
            for degree in degs:
                t1 = math.radians(degree)
                if 'x=' in vid_prefix:
                    y_rotation_matrix = get_rotation_matrix_y(t1)
                    z_rotation_matrix = get_rotation_matrix_z(t1)

                    aug_y_frames = collect_aug_rot(frames, y_rotation_matrix)
                    aug_z_frames = collect_aug_rot(frames, z_rotation_matrix)

                    # torch.save(aug_y_frames, f'{preprocess_folder}/{vid_prefix}_rot_y={degree}.pt')
                    # torch.save(aug_z_frames, f'{preprocess_folder}/{vid_prefix}_rot_z={degree}.pt')
                elif 'y=' in vid_prefix:
                    x_rotation_matrix = get_rotation_matrix_x(t1)
                    z_rotation_matrix = get_rotation_matrix_z(t1)

                    aug_x_frames = collect_aug_rot(frames, x_rotation_matrix)
                    aug_z_frames = collect_aug_rot(frames, z_rotation_matrix)

                    # torch.save(aug_x_frames, f'{preprocess_folder}/{vid_prefix}_rot_x={degree}.pt')
                    # torch.save(aug_z_frames, f'{preprocess_folder}/{vid_prefix}_rot_z={degree}.pt')
                elif 'z=' in vid_prefix:
                    x_rotation_matrix = get_rotation_matrix_x(t1)
                    y_rotation_matrix = get_rotation_matrix_y(t1)

                    aug_x_frames = collect_aug_rot(frames, x_rotation_matrix)
                    aug_y_frames = collect_aug_rot(frames, y_rotation_matrix)

                    # torch.save(aug_x_frames, f'{preprocess_folder}/{vid_prefix}_rot_x={degree}.pt')
                    # torch.save(aug_y_frames, f'{preprocess_folder}/{vid_prefix}_rot_y={degree}.pt')
                else:
                    print(vid_prefix)
                    print('Skipping this video')
        '''
    
    end_time = time.time()
    print("\nTotal Matrix Augmentation Time (s): ", end_time - start_time)

main()