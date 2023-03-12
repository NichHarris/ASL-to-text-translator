'''
Data augmentation - alter existing training data to produce new artificial data
-> prevents overfitting 
-> increases training set size
-> improves model performance
-- Geometric transforms: crop, rotate and translate
-- Color space transforms, kernel filters and noise injection (Not used due to using hand and pose estimation over images)
-- Speed change, perspective skewing, elastic distortions, rotating, shearing, cropping, mirroring
https://neptune.ai/blog/data-augmentation-in-python


Video augmentation of sign language videos
- Create more videos using vidaug to produce a more robust deep learning model
https://github.com/okankop/vidaug

- Horizontal flip on each augmented video (x2)
    -> For both left and right hand signers
Execution Time: 2m for 20 signs

Old 
- Spatial shear in X, Y by +/- 0.1 (+3)
    -> For different viewing angles and signer body position wrt video camera
- Random translate in x and y by +/- 50 pixels (+6)
    -> For shifting height and width positions of signers wrt video camera
- Crop video from each corner and center location to specific dimensions (5*3)
    -> For shifting depth position of signers wrt video camera
- Augmentation Factor: 1 (Horizontal flip) + [ 3 (Spatial shear) + 6 (Translation in x, y) + 5 (Crop pos in tl, tr, bl, br, c) * 3 (Crop factor of 0.925, 0.95, 0.975) ] * 2 (Horizontal flip on aug videos)
    = 49 times per video!
- Execution Time: 9m 12s for 5 gestures
'''

# pip3.10 install opencv-python scikit-image vidaug 

import os 
import cv2
import time
import numpy as np
from vidaug import augmentors as va

DATA_PATH = "../../../inputs/raw"

# Convert video to frames numpy array
def store_frames(video):
    # Open sign language video file
    cap = cv2.VideoCapture(video)

    # Calculate frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    # Close sign language video file
    cap.release()

    # Convert frames to numpy
    return np.array(frames), fps


# Pass augmentation sequence and file name
def augment_video(seq, frames, curr_folder, vid_name, fps, transform='flip'):
    # Augment frames
    aug_vid = seq(frames)
    height, width, _ = aug_vid[0].shape

    # Save augmented data
    # Define codec and video writer (Four character code for uniquely identifying file formats)
    fourcc = 'mp4v'
    video_writer = cv2.VideoWriter_fourcc(*fourcc)

    # Save video 
    out = cv2.VideoWriter(f"{DATA_PATH}/{curr_folder}/{vid_name}_{transform}.mp4", video_writer, fps, (width, height))
    for frame in aug_vid:
        out.write(frame)
    out.release()

    return aug_vid

def main():
    start_time = time.time()

    # Define flip sequence
    flip_seq = va.Sequential([ va.HorizontalFlip() ])

    # Get all actions/gestures names
    actions = os.listdir(DATA_PATH)
    for action in actions:
        print(f"\n-- Starting {action} augmentation --")

        # Get all filenames for augmentating each
        videos = os.listdir(f"{DATA_PATH}/{action}")

        # Augment video by video
        for video in videos:
            # Skip flipping if flipped videos
            if '_flip' in video:
                continue

            # Skip flipping if file already exists
            vid_name, *_ = video.split('.mp4')
            if os.path.isfile(f'{DATA_PATH}/{action}/{vid_name}_flip.mp4'):
                print(f"- Skipping: {vid_name} -")
                continue
            
            print(f"- {vid_name} -")

            # Capture frames per video
            frames, fps = store_frames(f"{DATA_PATH}/{action}/{video}")

            # Augment video using frames
            augment_video(flip_seq, frames, action, vid_name, fps)
        
    end_time = time.time()
    print("\nTotal Video Augmentation Time (s): ", end_time - start_time)

main()

'''
# IP: Potential video cropping augmentation
def zoom_translation_aug(frames, action, vid_name, fps):
    height, width, _ = frames[0].shape

    # Aug count: 4 (+2 (2*1) x 2)
    # Crop video from center to specific dimensions
    factors = [0.95, 0.975]
    # [0.925, 0.95, 0.975]
    for factor in factors:
        center_crop_seq = va.Sequential([ va.CenterCrop(size=(int(height * factor), int(width * factor))) ])
        aug_frames = augment_video(center_crop_seq, frames, action, vid_name, fps, 'crop_cc')

    # Aug count: 20 (+8 (4*2) x2)
    # Crop video from specific corner to specific dimensions
    corners = ['tl', 'tr', 'bl', 'br']
    for corner in corners:
        for factor in factors:
            corner_crop_seq =  va.Sequential([ va.CornerCrop(size=(int(height * factor), int(width * factor )), crop_position=corner) ])
            aug_frames = augment_video(corner_crop_seq, frames, action, vid_name, fps, f'crop_{corner}')

def cropping_script():
    start_time = time.time()

    # Get all actions/gestures names
    actions = os.listdir(DATA_PATH)
    for action in actions:
        print(f"\n-- Starting {action} augmentation --")

        # Get all filenames for augmentating each
        videos = os.listdir(f"{DATA_PATH}/{action}")

        # Augment video by video
        for video in videos:
            # Skip cropping if cropped videos
            if '_crop' in video:
                continue

            # Skip cropping if file already exists
            vid_name, *_ = video.split('.mp4')
            if os.path.isfile(f'{DATA_PATH}/{action}/{vid_name}_crop_cc.mp4'):
                print(f"- Skipping: {vid_name} -")
                continue
            
            print(f"- {vid_name} -")

            # Capture frames per video
            frames, fps = store_frames(f"{DATA_PATH}/{action}/{video}")

            # Augment video using frames
            zoom_translation_aug(frames, action, vid_name, fps)

            end_time = time.time()
            print("\nTotal Video Augmentation Time (s): ", end_time - start_time)
'''