'''
Fit data instances to 48 frames for training and testing neural network
- Create new directory to store dataset
- Load each data instance and compute number of frames
- Determine 

Execution Time: 13s for 5 words
New Execution Time: 201s for 5 words

Latest Execution Time: 72s for 10 words
'''

import time
import random
import torch
import math
from os import listdir, path, makedirs

DATASET_PATH = "../../../dataset_me"
PREPROCESS_PATH = "../../../preprocess-me"

INPUT_SIZE = 226
NUM_SEQUENCES = 48

def main():
    start_time = time.time()
    if not path.exists(DATASET_PATH):
        makedirs(DATASET_PATH)

    # Get all actions/gestures names
    actions = listdir(PREPROCESS_PATH)
    for action in actions:
        print(f"\n--- Starting {action} temporal fit ---")

        # Get all filenames for temporal fit to 48 frames
        videos = listdir(f"{PREPROCESS_PATH}/{action}")

        # Augment video by video
        for video in videos:
            print(f"\n-- Temporal fit video {video} --")

            # Load data instance
            frames = torch.load(f"{PREPROCESS_PATH}/{action}/{video}")
            
            # Calculate num frames over or under data frames input limit 
            num_frames = len(frames)
            missing_frames = NUM_SEQUENCES - num_frames

            if missing_frames == 0:
                print(f'Data already fitted to {NUM_SEQUENCES} frames')
                continue
            
            is_over_limit = missing_frames < 0
            print(f'Problem: {num_frames}')

            # Must select frames manually 
            missing_frames = abs(missing_frames)

            # Calculate num of times each frame and update frame population to properly sample missing frames
            frame_pop = range(num_frames)
            if num_frames == 0:
                # No frames in file
                continue

            if num_frames < missing_frames:
                factor = math.ceil(NUM_SEQUENCES / num_frames)
                frame_pop = list(frame_pop) * factor
            
            # Pick frames randomly to remove or duplicate based on data size
            frame_indices = sorted(random.sample(frame_pop, missing_frames), reverse=True)
            print(frame_indices)

            # Data temporal fit
            if is_over_limit:
                # Delete frames over limit
                for frame_index in frame_indices:
                    frames.pop(frame_index)
            else:
                # Duplicate missing frames
                for frame_index in frame_indices:
                    curr_frame = frames[frame_index]
                    frames.insert(frame_index, curr_frame)
            
            # Adjust format to torch tensors
            torch_frames = torch.zeros([NUM_SEQUENCES, INPUT_SIZE], dtype=torch.float)
            for seq, frame in enumerate(frames):
                torch_frames[seq] = frame
            
            # Save updated frames
            print(f'Fixed: {len(frames)}')
            torch.save(torch_frames, f'{DATASET_PATH}/{action}_{video}')

    end_time = time.time()
    print("\nTotal Data Temporal Fit Time (s): ", end_time - start_time)

main()
