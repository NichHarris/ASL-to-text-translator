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
'''

import os
import math
import time
import torch
import random

PREPROCESS_PATH = "../../../inputs/interim"

MM_RESHAPE = (67, 3)
NN_RESHAPE = (-1, )

def apply_rotation(coordinates, rotation_matrix):
    # [x, y, z] * rotation_matrix
    return torch.mm(coordinates, rotation_matrix)

def collect_aug_frame(frames, rotation_matrix):
    aug_frames = []

    # Perform augmentation
    for i, frame in enumerate(frames):
        # Detach frame to make changes and reshape for quick matrix multiplication
        aug_frame = frame.detach().clone()
        aug_frame = torch.reshape(aug_frame, MM_RESHAPE)

        aug_frame = apply_rotation(torch.reshape(aug_frame, MM_RESHAPE), rotation_matrix)
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

def load_sign_video(preprocess_folder, video):
    return torch.load(f"{preprocess_folder}/{video}")

def main():
    start_time = time.time()

    # Rotation degrees
    degrees = [-3, -2, -1, 1, 2, 3]

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
            if '_rot_' in vid_prefix or os.path.exists(f"{PREPROCESS_PATH}/{action}/{vid_prefix}_rot_x=1.pt"):
                continue
                        
            # Load data instance
            frames = load_sign_video(preprocess_folder, video)
            
            # Perform rotation with multiple degrees
            for degree in degrees:
                theta = math.radians(degree)

                # Define rotation matrix constants
                x_rotation_matrix = get_rotation_matrix_x(theta)
                y_rotation_matrix = get_rotation_matrix_y(theta)
                z_rotation_matrix = get_rotation_matrix_z(theta)

                # Augment frames using rotation matrices
                aug_x_frames = collect_aug_frame(frames, x_rotation_matrix)
                aug_y_frames = collect_aug_frame(frames, y_rotation_matrix)
                aug_z_frames = collect_aug_frame(frames, z_rotation_matrix)
      
                # Save augmented frames as torch files
                torch.save(aug_x_frames, f'{preprocess_folder}/{vid_prefix}_rot_x={degree}.pt')
                torch.save(aug_y_frames, f'{preprocess_folder}/{vid_prefix}_rot_y={degree}.pt')
                torch.save(aug_z_frames, f'{preprocess_folder}/{vid_prefix}_rot_z={degree}.pt')

    end_time = time.time()
    print("\nTotal Matrix Augmentation Time (s): ", end_time - start_time)

main()