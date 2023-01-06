'''
Perform video augmentation using rotation matrices and mediapipe coordinates
- Augmentation performed only on original video and horizontally flipped video 
    -> Ensure equal augmentation for left and right handed signers
- Small angle rotations in x, y and z directions
    -> Augmentation Factor: 3 (1, 2, 5 degrees) * 2 (+/- degrees) * 3 (x, y, z rotation) * 2 (original, horizontal aug) = 36 times per video!

- Execution Time: 1m 30s for 5 signs
'''

import numpy as np
import torch
from math import pi, sin, cos
from os import listdir, makedirs, path
import time

PREPROCESS_PATH = "./preprocess"
POSE_START = 21*3*2

# Convert degree to radians
degrees = [-1, -2, -5, 1, 2, 5]

def apply_rotation(coordinates, rotation_matrix):
    # rotation_matrix * [[x], [y], [z]]
    # [x, y, z] * rotation_matrix
    return torch.matmul(coordinates, rotation_matrix)

def collect_aug_frame(hands, pose, rotation_matrix):
    # Augmented frames
    aug_hands, aug_pose = hands.detach().clone(), pose.detach().clone()

    # Perform augmentation
    curr_ind = 0
    h_iter = iter(aug_hands)
    for x, y, z in zip(h_iter, h_iter, h_iter):
        # aug_hands = torch.cat((aug_hands, apply_rotation(torch.tensor([x, y, z]), rotation_matrix)))
        aug_hands[curr_ind:curr_ind + 3] = apply_rotation(torch.tensor([x, y, z]), rotation_matrix)
        curr_ind += 3
    
    curr_ind = 0
    p_iter = iter(aug_pose)
    for x, y, z, v in zip(p_iter, p_iter, p_iter, p_iter):
        # aug_pose = torch.cat((aug_pose, apply_rotation(torch.tensor([x, y, z]), rotation_matrix), torch.tensor([v])))
        aug_pose[curr_ind:curr_ind + 3] = apply_rotation(torch.tensor([x, y, z]), rotation_matrix)
        aug_pose[curr_ind + 3] = v
        curr_ind += 4

    return torch.cat((aug_hands, aug_pose))

def main():
    start_time = time.time()

    # Perform all degree rotations in each axis
    for i, degree in enumerate(degrees):
        print(f"\n--- Starting {degree} matrix rotation ---")
        theta = degree*pi/180

        # Define rotation matrix constants
        z_rotation_matrix = torch.tensor([ 
                [ cos(theta), -sin(theta), 0 ], 
                [ sin(theta), cos(theta), 0 ], 
                [ 0, 0, 1 ]
            ])

        x_rotation_matrix = torch.tensor([ 
                [ 1, 0, 0 ], 
                [ 0, cos(theta), -sin(theta) ], 
                [ 0, sin(theta), cos(theta) ]
            ])

        y_rotation_matrix = torch.tensor([ 
                [ cos(theta), 0, sin(theta) ], 
                [ 0, 1, 0], 
                [ -sin(theta), 0, cos(theta)]
            ])

        # Get all actions/gestures names
        actions = listdir(PREPROCESS_PATH)
        for action in actions:
            print(f"\n--- Starting {action} matrix augmentation ---")
            preprocess_folder = f"{PREPROCESS_PATH}/{action}"

            # Get all filenames for preprocessing each
            videos = listdir(preprocess_folder)
            for video in videos:
                vid_prefix = video.split(".")[0]
                vid_name, aug_num, *_ = vid_prefix.split("_")

                # Skip already augmented files except for horizontally flipped videos (left and right hand equally augmented)
                if vid_name == "mat" or (vid_name == "aug" and int(aug_num) % 49 != 1):
                    continue
                
                # Load data instance
                print(f"\n-- Matrix augmentation on {vid_prefix} --")
           
                frames = torch.load(f"{preprocess_folder}/{video}")
                
                # Store augmentated frames
                x_frames = []
                y_frames = []
                z_frames = []
                
                # Iterate frame by frame
                for frame in frames:
                    # Split tensor into tracked parts
                    hands = frame[0 : POSE_START]
                    pose = frame[POSE_START:]

                    # Augment frame using rotation matrices
                    x_aug_frame = collect_aug_frame(hands, pose, x_rotation_matrix)
                    x_frames.append(x_aug_frame)

                    y_aug_frame = collect_aug_frame(hands, pose, y_rotation_matrix)
                    y_frames.append(y_aug_frame)

                    z_aug_frame = collect_aug_frame(hands, pose, z_rotation_matrix)                    
                    z_frames.append(z_aug_frame)
                
                # Save augmented frames as torch files
                torch.save(x_frames, f'{preprocess_folder}/mat_x_{i}.pt')
                torch.save(y_frames, f'{preprocess_folder}/mat_y_{i}.pt')
                torch.save(z_frames, f'{preprocess_folder}/mat_z_{i}.pt')
    
    end_time = time.time()
    print("\nTotal Matrix Augmentation Time (s): ", end_time - start_time)

main()
