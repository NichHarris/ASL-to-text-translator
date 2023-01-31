import numpy as np
import torch
from math import pi, sin, cos
from os import listdir, makedirs, path, remove
import time
import random
import os

import torch.nn as nn
import re
import shutil

OLD_DATASET_PATH = './preprocess-1'
NEW_DATASET_PATH = './process'
POSE_START = 21*3*2

def extract_hands_pose(frame):
    # Return (hands, pose) tuple for each frame
    return (frame[0 : POSE_START], frame[POSE_START:])

def apply_rotation(coordinates, rotation_matrix):
    # rotation_matrix * [[x], [y], [z]]
    # [x, y, z] * rotation_matrix
    return torch.matmul(coordinates, rotation_matrix)

def collect_aug_frame(hands, pose):
    # Augmented frames
    aug_pose = torch.zeros([75])
    
    curr_ind = 0
    p_iter = iter(pose)
    for x, y, z, v in zip(p_iter, p_iter, p_iter, p_iter): 
        aug_pose[curr_ind:curr_ind + 3] = torch.tensor([x, y, z])
        curr_ind += 3

    return torch.cat((hands, aug_pose))

def load_sign_video(folder, video):
    return torch.load(f"{folder}/{video}")

def main():
    start_time = time.time()

    # Get all actions/gestures names
    files = listdir(OLD_DATASET_PATH)
    for fl in files:
        frames = load_sign_video(OLD_DATASET_PATH, fl)

        for i, frame in enumerate(frames):
            hands, pose = extract_hands_pose(frame)
            new_frame = collect_aug_frame(hands, pose)
            new_frames[i] = new_frame
    
        torch.save(new_frames, f'{NEW_DATASET_PATH}/{fl}')
    
    end_time = time.time()
    print("\nTotal Visibility Removal Time (s): ", end_time - start_time)


def copy_specific_files():
    start_time = time.time()

    # Get all actions/gestures names
    files = listdir(OLD_DATASET_PATH)
    for fl in sorted(files):
        # Double rotations
        if re.search(r'rot_.*_rot', fl):
            shutil.copyfile(f'{OLD_DATASET_PATH}/{fl}', f'{NEW_DATASET_PATH}/{fl}')

        # Original videos
        if not re.search(r'trans', fl) and not re.search(r'rot', fl):
            shutil.copyfile(f'{OLD_DATASET_PATH}/{fl}', f'{NEW_DATASET_PATH}/{fl}')
    
    end_time = time.time()
    print("\nTotal Augmentation Copy Time (s): ", end_time - start_time)


OLD_DATASET_PATH_1 = './dataset_48'
OLD_DATASET_PATH_2 = './dataset_me'
NEW_DATASET_PATH = './dataset_joint'
def join_dataset_files():
    start_time = time.time()
    
    # Get all actions/gestures names
    files1= listdir(OLD_DATASET_PATH_1)
    for fl in sorted(files1):
        shutil.copyfile(f'{OLD_DATASET_PATH_1}/{fl}', f'{NEW_DATASET_PATH}/{fl}')
    
    files2 = listdir(OLD_DATASET_PATH_2)
    for fl in sorted(files2):
        fl_prefix = fl.split('.pt')[0]
        shutil.copyfile(f'{OLD_DATASET_PATH_2}/{fl}', f'{NEW_DATASET_PATH}/{fl_prefix}_me.pt')

    end_time = time.time()
    print("\nTotal Dataset Join Time (s): ", end_time - start_time)

OLD_DATASET_PATH = './preprocess-1'
NEW_DATASET_PATH = './preprocess'
def move_specific_files():
    start_time = time.time()

    # Get all actions/gestures names
    actions = listdir(OLD_DATASET_PATH)
    for action in actions:
        files = sorted(listdir(f'{OLD_DATASET_PATH}/{action}'))
        inc = 0

        makedirs(f'{NEW_DATASET_PATH}/{action}')

        for fl in files:
            # Double rotations
            if re.search(r'rot_.*_rot', fl):
                shutil.move(f'{OLD_DATASET_PATH}/{action}/{fl}', f'{NEW_DATASET_PATH}/{action}/{fl}')
                inc += 1
            elif re.search(r'rot_.*', fl) and not re.search(r'trans', fl):
                shutil.move(f'{OLD_DATASET_PATH}/{action}/{fl}', f'{NEW_DATASET_PATH}/{action}/{fl}')
                inc += 1

            # Original videos
            if not re.search(r'trans', fl) and not re.search(r'rot', fl):
                shutil.move(f'{OLD_DATASET_PATH}/{action}/{fl}', f'{NEW_DATASET_PATH}/{action}/{fl}')
                inc += 1

    print(inc)

    end_time = time.time()
    # print("\nTotal Augmentation Copy Time (s): ", end_time - start_time)

# move_specific_files()
# 163 * word sign videos

CURR_FOLDER='./live_data'
def move_live_data():
    start_time = time.time()

    # Get all actions/gestures names
    curr_vid = ''
    videos = sorted(listdir(CURR_FOLDER))
    for video in videos:
        if not os.path.isfile(f'./{CURR_FOLDER}/{video}'):
            continue
        
        vid_name = video.split('_')[0]
        if curr_vid != vid_name:
            curr_vid = vid_name
            if not os.path.isdir(f'./{CURR_FOLDER}/{vid_name}'):
                os.makedirs(f'./{CURR_FOLDER}/{vid_name}')
        shutil.move(f'{CURR_FOLDER}/{video}', f'{CURR_FOLDER}/{curr_vid}/{video}')

    end_time = time.time()

TEST_FOLDER='./live_test'
def move_live_test():
    start_time = time.time()

    # Get all actions/gestures names
    curr_vid = ''
    actions = listdir(CURR_FOLDER)
    for action in actions:
        vid_path = f'./{CURR_FOLDER}/{action}'
        videos = listdir(vid_path)
        count=0
        for video in videos:
            if count >= 10:
                break
            shutil.move(f'{CURR_FOLDER}/{action}/{video}', f'{TEST_FOLDER}/{video}')
            count += 1
        
    end_time = time.time()

# 
CURR_FOLDER='./ali'
NEW_FOLDER='./live_ali_test'
def move_data():
    start_time = time.time()

    # Get all actions/gestures names
    curr_vid = ''
    actions = listdir(CURR_FOLDER)
    for action in actions:
        vid_path = f'{CURR_FOLDER}/{action}'
        videos = listdir(vid_path)
        for video in videos:
            if video.startswith(action):
                shutil.copyfile(f'{CURR_FOLDER}/{action}/{video}', f'{NEW_FOLDER}/{video}')
        
    end_time = time.time()

# move_data()

def delete_data():
    count = 0
    curr_vid = ''
    videos = sorted(os.listdir('./live_ali_test'))
    for video in videos:
        vid_name = video.split('_')[0]
        if curr_vid != vid_name:
            curr_vid = vid_name
            count = 0
        elif count >= 10:
            # print(f'./live_ali_test/{video}')
            os.remove(f'./live_ali_test/{video}')
        else:
            count += 1

CURR_FOLDER='./dataset_joint'
def move_orig_data():
    start_time = time.time()

    # Get all actions/gestures names
    curr_vid = ''
    
    videos = listdir(CURR_FOLDER)
    for video in videos:
        vid_word, vid_pref, *_ = video.split('_')
        if vid_word == vid_pref:
            shutil.copyfile(f'{CURR_FOLDER}/{video}', f'./dataset_only/{video}')
        
    end_time = time.time()

# move_orig_data()

import os
import time

# Decorator for providing runtime of script
def time_decorator(main_fn, output_msg):
    # Wrapper function called by decorator
    def wrapper():
        start_time = time.time()

        # Script method call
        main_fn()

        end_time = time.time()
        print(output_msg, end_time - start_time)

    return wrapper

def main():
    new_dataset_files = os.listdir('./dataset_48')

    linux_cmd = ''
    for i, new_file in enumerate(new_dataset_files):
        if i % 1000 == 0:
            linux_cmd += ';\ngit add '
        linux_cmd += f'./dataset2/{new_file} '

# test = time_decorator(main, 'Total time for git command generation: ')
# test()

# with open('./test.txt', 'w') as file:
#     file.write(linux_cmd)



# 1 landmarks detected, 0 none
frames1 = [0, 0, 0, 1, 1, 1, 1, 1, 1]
frames2 = [0, 0, 1, 1, 1, 1, 1, 1, 1]
frames3 = [0, 0, 1, 1, 0, 0, 1, 1, 1]


endframes1 = [1, 1, 1, 1, 1, 1, 1, 0, 0]
endframes2 = [1, 1, 1, 1, 0, 0, 1, 0, 0]
endframes3 = [1, 1, 1, 0, 0, 0, 0, 0, 0]




first_frame = bin_search(frames1, 1)
print(first_frame)
print(frames1)
print(frames1[first_frame:])

end_frame = bin_search(endframes2, 0)
print(end_frame)
print(endframes2)
print(endframes2[:end_frame])

# Search