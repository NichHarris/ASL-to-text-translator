'''
Perform video augmentation using rotation matrices and mediapipe coordinates
- Augmentation performed only on original video and horizontally flipped video 
    -> Ensure equal augmentation for left and right handed signers
- Small angle rotations in x, y and z directions
    -> Augmentation Factor: 3 (1, 2, 5 degrees) * 2 (+/- degrees) * 3 (x, y, z rotation) * 2 (original, horizontal aug) = 36 times per video!

- Execution Time: 1m 30s for 5 signs
'''

# TODO: Check if visibility needs to be slightly modified as well

import numpy as np
import torch
from math import pi, sin, cos
from os import listdir, makedirs, path
import time
import random

PREPROCESS_PATH = "./preprocess"
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
    for x, y, z, v in zip(p_iter, p_iter, p_iter, p_iter): #100 keypoints for i in range(24)
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

def get_rotated_frames(video, rot1_matrix, rot2_matrix): 
    frames = torch.load(f"{preprocess_folder}/{video}")

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

def save_rotated_frames(vid_prefix, rot1_frames, rot1_angle, rot2_frames, rot2_angle): 
    torch.save(rot1_frames, f'{preprocess_folder}/{vid_prefix}_{rot1_angle}.pt')
    torch.save(rot2_frames, f'{preprocess_folder}/{vid_prefix}_{rot2_angle}.pt')

def main():
    start_time = time.time()

    # Multiple Translation (on Original videos - 36x)
    shift_pcs = [-0.015, -0.01, -0.005, 0.005, 0.01, 0.015]

    ct = torch.tensor([1])
    frames = torch.load(f"{PREPROCESS_PATH}/bye/bye_0.pt")

    for dx in shift_pcs:
        for dy in shift_pcs:
            trans_matrix = get_translation_matrix(dx, dy)
            print(f'Shift x by: {dx}%, Shift y by: {dy}%, Shift z by: {dz}%')

            for frame in frames:
                hands, pose = extract_hands_pose(frame)

                # Augmented frames
                aug_hands, aug_pose = hands.detach().clone(), pose.detach().clone()

                # Perform augmentation
                # print(torch.cat((aug_hands, aug_pose)))
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

                # print(torch.cat((aug_hands, aug_pose)))
                # for i in range(25):
                #     curr_ind = i*4
                #     print(aug_pose[curr_ind:curr_ind + 3] / pose[curr_ind:curr_ind + 3])

    # Single Rotation (on Orig + Translated - 6x)
    degrees = [-5, -2, -1, 1, 2, 5]

    # Perform all degree rotations in each axis
    for i, degree in enumerate(degrees):
        # Convert degrees to radians
        print(f"\n--- Starting {degree} matrix rotation ---")
        theta = degree*pi/180

        # Define rotation matrix constants
        x_rotation_matrix = get_rotation_matrix_x(theta)
        y_rotation_matrix = get_rotation_matrix_y(theta)
        z_rotation_matrix = get_rotation_matrix_z(theta)

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

                    # TODO: Currently only applying same rotation angle for 2 rotations
                    # - Make rotations on all angles
                    # Double augmment frame in 2 diff rotations
                    # x_hands, x_pose = x_aug_frame[0 : POSE_START], x_aug_frame[POSE_START:]
                    # xy_aug_frame = collect_aug_frame(x_hands, x_pose, y_rotation_matrix)
                    # xy_frames.append(xy_aug_frame)

                    # y_hands, y_pose = y_aug_frame[0 : POSE_START], y_aug_frame[POSE_START:]
                    # yz_aug_frame = collect_aug_frame(y_hands, y_pose, z_rotation_matrix)
                    # yz_frames.append(yz_aug_frame)

                    # z_hands, z_pose = z_aug_frame[0 : POSE_START], z_aug_frame[POSE_START:]
                    # zx_aug_frame = collect_aug_frame(z_hands, z_pose, x_rotation_matrix)                    
                    # zx_frames.append(zx_aug_frame)
                
                # # Save augmented frames as torch files
                # torch.save(x_frames, f'{preprocess_folder}/mat_x_{i}.pt') # x={degree}
                # torch.save(y_frames, f'{preprocess_folder}/mat_y_{i}.pt')
                # torch.save(z_frames, f'{preprocess_folder}/mat_z_{i}.pt')

                # # Save double augmented frames as torch files
                # torch.save(xy_frames, f'{preprocess_folder}/mat_xy_{i}.pt')
                # torch.save(yz_frames, f'{preprocess_folder}/mat_yz_{i}.pt')
                # torch.save(zx_frames, f'{preprocess_folder}/mat_zx_{i}.pt')
    
    # Multiple Rotation (on Single Rotation - 16x)
    small_degrees = [-2, -1, 1, 2]
    for d1 in small_degrees:
        for d2 in small_degrees:
            t1 = d1*pi/180
            t2 = d2*pi/180

            vid_prefix = video.split(".")[0]
            if 'x=' in vid_prefix:
                y_rotation_matrix = get_rotation_matrix_y(t1)
                z_rotation_matrix = get_rotation_matrix_z(t2)

                y_aug_frames, z_aug_frames = get_rotated_frames(video, y_rotation_matrix, z_rotation_matrix)
                save_rotated_frames(vid_prefix, y_aug_frames, f'y={d1}', z_aug_frames, f'z={d2}')
            elif 'y=' in vid_prefix:
                x_rotation_matrix = get_rotation_matrix_x(t1)
                z_rotation_matrix = get_rotation_matrix_z(t2)

                x_aug_frames, z_aug_frames = get_rotated_frames(video, x_rotation_matrix, z_rotation_matrix)
                save_rotated_frames(vid_prefix, x_aug_frames, f'x={d1}', z_aug_frames, f'z={d2}')
            elif 'z=' in vid_prefix:
                x_rotation_matrix = get_rotation_matrix_x(t1)
                y_rotation_matrix = get_rotation_matrix_y(t2)

                x_aug_frames, y_aug_frames = get_rotated_frames(video, x_rotation_matrix, y_rotation_matrix)
                save_rotated_frames(vid_prefix, x_aug_frames, f'x={d1}', y_aug_frames, f'y={d2}')
            else:
                print(vid_prefix)
                print('Skipping this video')
    
    end_time = time.time()
    print("\nTotal Matrix Augmentation Time (s): ", end_time - start_time)

# main()



        # trans_frame = []
        # for frame in frames:
        #     # Split tensor into tracked parts
        #     hands = frame[0 : POSE_START]
        #     pose = frame[POSE_START:]

        #     # Augment frame using translation matrix
        #     xy_trans_frame = collect_aug_frame(hands, pose, xy_shift_matrix)

        # torch.save(xy_trans_frames, f'{preprocess_folder}/mat_trans_xy_{i}.pt')


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


# theta = -1*pi/180
# x_rotation_matrix = torch.tensor([ 
#     [ 1, 0, 0, 0], 
#     [ 0, cos(theta), -sin(theta), 0 ], 
#     [ 0, sin(theta), cos(theta), 0 ],
#     [ 0, 0, 0, 1 ]
# ])

# y_rotation_matrix = torch.tensor([ 
#     [ cos(theta), 0, sin(theta), 0], 
#     [ 0, 1, 0, 0], 
#     [ -sin(theta), 0, cos(theta), 0],
#     [ 0, 0, 0, 1 ]
# ])




# print("Before", new_test)
# test1 = apply_rotation(new_test, x_rotation_matrix)
# print(test1)
# test1_1 = apply_rotation(test1, y_rotation_matrix)
# print("After", test1_1)

# print("Before", new_test)
# test2 = apply_rotation(new_test, y_rotation_matrix)
# print(test2)
# test2_1 = apply_rotation(test2, x_rotation_matrix)
# print("After", test2_1)
