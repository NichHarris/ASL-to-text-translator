'''
...
Execution Time: 13s for 5 words
'''

import time
import random
import torch
import math
from os import listdir, path, makedirs

DATA_FRAMES_LIMIT = 48
PREPROCESS_PATH = "./preprocess"
DATASET_PATH = "./dataset"

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
            missing_frames = DATA_FRAMES_LIMIT - num_frames

            if missing_frames == 0:
                print("Data already fitted to 48 frames")
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
                factor = math.ceil(DATA_FRAMES_LIMIT / num_frames)
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
            
            # Save updated frames
            print(f'Fixed: {len(frames)}')
            torch.save(frames, f'{DATASET_PATH}/{action}_{video}')

    end_time = time.time()
    print("\nTotal Data Temporal Fit Time (s): ", end_time - start_time)

main()
