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

# TODO: Check if visibility needs to be slightly modified as well

import numpy as np
import torch
from math import pi, sin, cos
from os import listdir, makedirs, path, remove
import time
import random

PREPROCESS_PATH = "../../../preprocess-me"
POSE_START = 21*3*2

def extract_hands_pose(frame):
    # Return (hands, pose) tuple for each frame
    return (frame[0 : POSE_START], frame[POSE_START:])

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
    for x, y, z in zip(h_iter, h_iter, h_iter): # 126 keypoints - TODO: for i in range(41)
        # aug_hands = torch.cat((aug_hands, apply_rotation(torch.tensor([x, y, z]), rotation_matrix)))
        aug_hands[curr_ind:curr_ind + 3] = apply_rotation(torch.tensor([x, y, z]), rotation_matrix)
        curr_ind += 3
    
    curr_ind = 0
    p_iter = iter(aug_pose)
    for x, y, z, v in zip(p_iter, p_iter, p_iter, p_iter): #100 keypoints - TODO: for i in range(24)
        # aug_pose = torch.cat((aug_pose, apply_rotation(torch.tensor([x, y, z]), rotation_matrix), torch.tensor([v])))
        aug_pose[curr_ind:curr_ind + 3] = apply_rotation(torch.tensor([x, y, z]), rotation_matrix)
        aug_pose[curr_ind + 3] = v # TODO: Modify v
        curr_ind += 4

    return torch.cat((aug_hands, aug_pose))

# Return rotation matrix with theta
def get_rotation_matrix_x(theta):
    return torch.tensor([ 
                [ 1, 0, 0 ],
                [ 0, cos(theta), -sin(theta) ],
                [ 0, sin(theta), cos(theta) ]
            ])

def get_rotation_matrix_y(theta):
    return torch.tensor([ 
                [ cos(theta), 0, sin(theta) ], 
                [ 0, 1, 0], 
                [ -sin(theta), 0, cos(theta)]
            ])

def get_rotation_matrix_z(theta):
    return torch.tensor([ 
                [ cos(theta), -sin(theta), 0 ], 
                [ sin(theta), cos(theta), 0 ], 
                [ 0, 0, 1 ]
            ])

# Return translation matrix with dx, dy, dz
def get_translation_matrix(dx, dy):
    # Create minimal shift in z axis
    dz = 0.0005 * random.choice([-1, 1])
    return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [dx, dy, dz, 1]
        ])

def get_rotated_frames(frames, rot1_matrix, rot2_matrix): 
    rot1_frames, rot2_frames = [], []

    # Iterate frame by frame
    for frame in frames:
        # Split tensor into tracked parts
        hands = frame[0 : POSE_START]
        pose = frame[POSE_START:]

        # Augment frame using rotation matrices
        rot1_frame = collect_aug_frame(hands, pose, rot1_matrix)
        rot1_frames.append(rot1_frame)

        rot2_frame = collect_aug_frame(hands, pose, rot2_matrix)
        rot2_frames.append(rot2_frame)
    
    return rot1_frames, rot2_frames

def load_sign_video(preprocess_folder, video):
    return torch.load(f"{preprocess_folder}/{video}")


def save_rotated_frames(preprocess_folder, vid_prefix, rot1_frames, rot1_angle, rot2_frames, rot2_angle): 
    torch.save(rot1_frames, f'{preprocess_folder}/{vid_prefix}_rot_{rot1_angle}.pt')
    torch.save(rot2_frames, f'{preprocess_folder}/{vid_prefix}_rot_{rot2_angle}.pt')

def main():
    start_time = time.time()

    shift_pcs = [-0.015, -0.01, -0.005, 0.005, 0.01, 0.015]
    degrees = [-3, -2, -1, 1, 2, 3]
    small_degrees = [-2, -1, 1, 2]
    ct = torch.tensor([1])

    # Get all actions/gestures names
    actions = listdir(PREPROCESS_PATH)
    for action in actions:
        print(f"\n--- Starting {action} matrix augmentation ---")
        preprocess_folder = f"{PREPROCESS_PATH}/{action}"
        '''
        # Get all filenames for preprocessing each
        print(f'\n-- Performing Simple Translations --')
        videos = listdir(preprocess_folder)
        for video in videos:   
            video_prefix = video.split('.pt')[0]
            sign_name, video_no, *_ = video_prefix.split('_')

            frames = load_sign_video(preprocess_folder, video)

            # Multiple Translation (on Original videos - 36x)
            for dx in shift_pcs:
                for dy in shift_pcs:
                    trans_matrix = get_translation_matrix(dx, dy)

                    aug_frames = []
                    for frame in frames:
                        hands, pose = extract_hands_pose(frame)

                        # Augmented frames
                        aug_hands, aug_pose = hands.detach().clone(), pose.detach().clone()

                        # Perform augmentation
                        curr_ind = 0
                        for i in range(42):
                            curr_ind = i*3

                            # Skip undetected hands
                            if aug_hands[curr_ind] == 0 and aug_hands[curr_ind + 1] == 0 and aug_hands[curr_ind + 2] == 0:
                                continue

                            prep_hands = torch.cat([ aug_hands[curr_ind:curr_ind + 3], ct ])
                            aug_hands[curr_ind:curr_ind + 3] = torch.matmul(prep_hands, trans_matrix)[0:3]

                        for i in range(25):
                            curr_ind = i*4
                            prep_pose = torch.cat([ aug_pose[curr_ind:curr_ind + 3], ct ])
                            aug_pose[curr_ind:curr_ind + 3] = torch.matmul(prep_pose, trans_matrix)[0:3]
                        
                        aug_frames.append(torch.cat((aug_hands, aug_pose)))

                    torch.save(aug_frames, f'{preprocess_folder}/mat_{video_no}_trans_x={dx}_y={dy}.pt')  
        '''

        print(f'\n-- Performing Single Rotations --')
        videos = listdir(preprocess_folder)
        for video in videos:
            vid_prefix = video.split('.pt')[0]
            sign_name, video_no, *_ = vid_prefix.split('_') 

            frames = load_sign_video(preprocess_folder, video)

            # Single Rotation (on Orig + Translated - 6x * 3x)
            for degree in degrees:
                theta = degree*pi/180

                # Define rotation matrix constants
                x_rotation_matrix = get_rotation_matrix_x(theta)
                y_rotation_matrix = get_rotation_matrix_y(theta)
                z_rotation_matrix = get_rotation_matrix_z(theta)

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
                
                # # Save augmented frames as torch files                
                aug_prefix = f'mat_{video_no}'
                if vid_prefix.startswith('mat_'):
                    aug_prefix = vid_prefix

                torch.save(x_frames, f'{preprocess_folder}/{aug_prefix}_rot_x={degree}.pt')
                torch.save(y_frames, f'{preprocess_folder}/{aug_prefix}_rot_y={degree}.pt')
                torch.save(z_frames, f'{preprocess_folder}/{aug_prefix}_rot_z={degree}.pt')

        print(f'\n-- Performing Multiple Rotations --')
        videos = listdir(preprocess_folder)
        for video in videos:   
            vid_prefix = video.split('.pt')[0]
            sign_name, video_no, *_ = vid_prefix.split('_') 

            if not vid_prefix.startswith('mat_') or '_trans_' in vid_prefix:
                continue

            print(vid_prefix)
            frames = torch.load(f"{preprocess_folder}/{video}")

            # Multiple Rotation (on Single Rotation - 4x)
            for d1 in small_degrees:
                t1 = d1*pi/180

                if 'x=' in vid_prefix:
                    y_rotation_matrix = get_rotation_matrix_y(t1)
                    z_rotation_matrix = get_rotation_matrix_z(t1)

                    y_aug_frames, z_aug_frames = get_rotated_frames(frames, y_rotation_matrix, z_rotation_matrix)
                    save_rotated_frames(preprocess_folder, vid_prefix, y_aug_frames, f'y={d1}', z_aug_frames, f'z={d1}')
                elif 'y=' in vid_prefix:
                    x_rotation_matrix = get_rotation_matrix_x(t1)
                    z_rotation_matrix = get_rotation_matrix_z(t1)

                    x_aug_frames, z_aug_frames = get_rotated_frames(frames, x_rotation_matrix, z_rotation_matrix)
                    save_rotated_frames(preprocess_folder, vid_prefix, x_aug_frames, f'x={d1}', z_aug_frames, f'z={d1}')
                elif 'z=' in vid_prefix:
                    x_rotation_matrix = get_rotation_matrix_x(t1)
                    y_rotation_matrix = get_rotation_matrix_y(t1)

                    x_aug_frames, y_aug_frames = get_rotated_frames(frames, x_rotation_matrix, y_rotation_matrix)
                    save_rotated_frames(preprocess_folder, vid_prefix, x_aug_frames, f'x={d1}', y_aug_frames, f'y={d1}')
                else:
                    print(vid_prefix)
                    print('Skipping this video')
    
    end_time = time.time()
    print("\nTotal Matrix Augmentation Time (s): ", end_time - start_time)

main()

# videos = listdir('./preprocess/bye')
# removed_videos = []
# kept_videos = []
# for video in videos:
#     if 'bye' in video:
#         kept_videos.append(video)
#     else:
#         removed_videos.append(video)

# for removed in removed_videos:
#     remove(f'./preprocess/bye/{removed}')


# x_rotation_matrix = get_rotation_matrix_x(-3*pi/180)
# frames = torch.load(f"{PREPROCESS_PATH}/bye/bye_0.pt")

# # Store augmentated frames
# x_frames = []

# # Iterate frame by frame
# for frame in frames:
#     # Split tensor into tracked parts
#     hands = frame[0 : POSE_START]
#     pose = frame[POSE_START:]
    
#     x_aug_frame = collect_aug_frame(hands, pose, x_rotation_matrix)
#     x_frames.append(x_aug_frame)

#     for i in range(25):
#         curr_ind = 126 + i*4
#         print(frame[curr_ind:curr_ind+3] / x_aug_frame[curr_ind:curr_ind+3])





'''
Angle: +/- (1, 2, 5)
Percent: +/- (1, 2, 3)

Matrix Augmentation
- Rotation (3*6)*(4*1) #Consider which angle was used in rotation and axis  
    - Single: Rx, Ry, Rz
    - Multiple: Rx + Ry, Ry + Rz, Rz + Rx
- Translation 
    - Single (Combine with Rotation): Lx, Rx, Uy, Dy
        - Maybe: Fz, Bz
    - Multiple: Lx + Uy, Rx + Uy, Lx + Dy, Rx + Dy
        - Maybe: Lx + Fz, Rx + Fz, Lx + Bz, Rx + Bz
- Scale
    - Single: Zoom in, zoom out 
        - Note: Adjust z coordinates plus scale
    - Multiple: ...

Steps:
- Rotate in x, y, z
- Translate all video in x and y, y and z, z and x



shift_pc = 0.01
(1 - shift_pc) # Shift left, or down
(1 + shift_pc) # Shift right, or up


# Scale



# Idea: Can perform testing on videos we record to avoid manually testing 


'''