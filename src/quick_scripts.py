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
'''
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
'''

# folder = '../inputs/raw-2'
# actions = os.listdir(folder)
# for action in actions:
#     os.rename(f'{folder}/{action}', f'{folder}/{action.upper()}')

'''
import shutil

word_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'bad': 36, 'bye': 37, 'easy': 38, 'good': 39, 'happy': 40, 'hello': 41, 'how': 42, 'like': 43, 'me': 44, 'meet': 45, 'more': 46, 'no': 47, 'please': 48, 'sad': 49, 'she': 50, 'sorry': 51, 'thank you': 52, 'want': 53, 'what': 54, 'when': 55, 'where': 56, 'which': 57, 'who': 58, 'why': 59, 'yes': 60, 'you': 61}
# {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'bad', 37: 'bye', 38: 'easy', 39: 'good', 40: 'happy', 41: 'hello', 42: 'how', 43: 'like', 44: 'me', 45: 'meet', 46: 'more', 47: 'no', 48: 'please', 49: 'sad', 50: 'she', 51: 'sorry', 52: 'thank you', 53: 'want', 54: 'what', 55: 'when', 56: 'where', 57: 'which', 58: 'who', 59: 'why', 60: 'yes', 61: 'you'}
signs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bad', 'bye', 'easy', 'good', 'happy', 'hello', 'how', 'like', 'me', 'meet', 'more', 'no', 'please', 'sad', 'she', 'sorry', 'thank you', 'want', 'what', 'when', 'where', 'which', 'who', 'why', 'yes', 'you']
for i, sign in enumerate(signs):
    word_dict[sign] = i
print(word_dict)

# Missing how

ali_actions = os.listdir('../../../tests/live_data_aug_ali')
count = 0
for action in ali_actions:
    is_present = word_dict.get(action)
    if is_present:
        print(action)
        # shutil.copytree(f'../../../tests/live_data_aug_ali/{action}', f'../../../tests/ali_top_62/{action}', False, None)
        count += 1
'''
'''
BASE_DIR = '../../../tests/ali_top_62'
ali_actions = os.listdir(BASE_DIR)
for action in ali_actions:
    VID_DIR = f'{BASE_DIR}/{action}'
    ali_videos = os.listdir(VID_DIR)
    for video in ali_videos:
        frames = torch.load(f'{VID_DIR}/{video}')
        torch_frames = torch.zeros([48, 201], dtype=torch.float)
        for seq, frame in enumerate(frames):
            torch_frames[seq] = frame
        
        torch.save(torch_frames, f'../../../tests/top_62_ali/{video}')
'''

# print(len(os.listdir('../../../tests/top_62_ali')))
'''
INTERIM_DIR = '../../../inputs/interim'
interim_actions = os.listdir(INTERIM_DIR)
for action in interim_actions:
    if os.path.isdir(f'{INTERIM_DIR}/{action}'):
        interim_vids = os.listdir(f'{INTERIM_DIR}/{action}')
        for vid in interim_vids:
            if '_scale' in vid or '_rot' in vid:
                continue

            print(f'"inputs/interim/{action}/{vid}"', end=" ")
print("\n")
'''

# INTERIM_DIR = '../../../inputs/interim-2'
# interim_actions = os.listdir(INTERIM_DIR)
# for action in interim_actions:
#     if os.path.isdir(f'{INTERIM_DIR}/{action}'):
#         interim_vids = os.listdir(f'{INTERIM_DIR}/{action}')
#         for vid in interim_vids:
#             if '_scale' in vid or '_rot' in vid:
#                 continue

#             print(f'inputs/interim-2/{action}/{vid}', end=" ")



'''
BASE_DIR = '../../../tests/top_62_ali'
ali_vids = os.listdir(BASE_DIR)
for vid in ali_vids:
    # if ' 2_' in vid:
    #   os.rename(f'{BASE_DIR}/{vid}', f'{BASE_DIR}/{vid.replace(" 2_", "_")}')
    
    # os.rename(f'{BASE_DIR}/{vid}', f'{BASE_DIR}/{vid.replace(".pt", "_ali.pt")}')
'''

'''
BASE_DIR = '../../../tests/live_data_aug_ali'
ali_actions = os.listdir(BASE_DIR)
for action in ali_actions:
    ali_videos = os.listdir(f'{BASE_DIR}/{action}')
    for video in ali_videos:
        frames = torch.load(f'{BASE_DIR}/{action}/{video}')
        new_frames = []
        for frame in frames:
            new_frame = torch.zeros([201])
            new_frame[:126] = frame[:126]
            pose = frame[126:]
            for i in range(25):
                new_frame[126 + i*3:126 + (i + 1)*3] = frame[126 + i*4:126 + (i + 1)*4 - 1]
            
            new_frames.append(new_frame)
        
        torch.save(new_frames, f'{BASE_DIR}/{action}/{video}')
'''
'''
for action in os.listdir('../inputs/interim'):
    i = 0
    total = 0
    for vid in os.listdir(f'../inputs/interim/{action}'):
        if '_scale' not in vid and '_rot' not in vid:
            if '_T' in vid or '_R' in vid or '_D' in vid or '_S' in vid:
                print(f'inputs/interim/{action}/{vid}', end=" ")
                i+=1
            total += 1
        
    # print(i, total,  "\n")
'''