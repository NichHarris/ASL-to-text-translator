'''
https://github.com/okankop/vidaug
Vidaug - video augmentation for deep learning

Data augmentation - alter existing training data to produce new artificial data
-> prevents overfitting 
-> increases training set size
-> improves model performance
-- Geometric transforms: crop, rotate and translate
-- Color space transforms, kernel filters and noise injection (Not used due to using hand and pose estimation over images)
-- Speed change, perspective skewing, elastic distortions, rotating, shearing, cropping, mirroring
https://neptune.ai/blog/data-augmentation-in-python
'''

# pip3.10 install opencv-python scikit-image vidaug 

import cv2
import time
import numpy as np
from os import listdir
from vidaug import augmentors as va

AUG_FACTOR = 49
DATA_PATH = "./data"

# Get execution time per augmentation
def provide_aug_update(aug_type, aug_ind):
    # aug_time  = time.time() - aug_start
    print(f"{aug_type} - {((aug_ind - 1) % AUG_FACTOR) + 1} / {AUG_FACTOR}")

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

    # Convert frames to numpy
    return np.array(frames), fps


# Pass augmentation sequence and file name
def augment_video(seq, frames, curr_folder, aug_num, fps):
    # Augment frames
    aug_vid = seq(frames)
    height, width, _ = aug_vid[0].shape

    # # Save augmented data
    # # Define codec and video writer (Four character code for uniquely identifying file formats)
    # fourcc = 'mp4v'
    # video_writer = cv2.VideoWriter_fourcc(*fourcc)

    # # Save video 
    # out = cv2.VideoWriter("./test/tf-1.mp4", video_writer, fps, (width, height)) #f"{DATA_PATH}/{curr_folder}/aug_{aug_num}.mp4"
    # for frame in aug_vid:
    #     out.write(frame)
    # out.release()

    return aug_vid

# Perform augmentation 
def augmentation_sequences(aug_count, frames, action, fps):
    i = aug_count
    _, height, width, _ = frames.shape

    # Aug count: 1 (+1)
    flip_seq = va.Sequential([ va.HorizontalFlip() ])
    
    i+=1
    augment_video(flip_seq, frames, action, i, fps)
    provide_aug_update("Flip", i)

    # Aug count: 7 (+3 x 2) 
    # Spatial shear in X, Y
    shear_seq = va.Sequential([ va.RandomShear(x=0.1, y=0.1) ])
    for _ in range(0, 3):
        i += 1
        aug_frames = augment_video(shear_seq, frames, action, i, fps)
        provide_aug_update("Shear", i)

        i += 1
        augment_video(flip_seq, aug_frames, action, i, fps)
        provide_aug_update(" + Flip", i)

    # Aug count: 19 (+6 x 2)
    # Translate in X, Y 
    translate_seq = va.Sequential([ va.RandomTranslate(x=50, y=50) ])
    for _ in range(0, 6):
        i += 1
        aug_frames = augment_video(translate_seq, frames, action, i, fps)
        provide_aug_update("Translation", i)

        i += 1
        augment_video(flip_seq, aug_frames, action, i, fps)
        provide_aug_update(" + Flip", i)

    # Aug count: 25 (+3 x 2)
    # Crop video from center to specific dimensions
    factors = [0.925, 0.95, 0.975]
    for factor in factors:
        center_crop_seq = va.Sequential([ va.CenterCrop(size=(int(height * factor), int(width * factor ))) ])
    
        i+=1    
        aug_frames = augment_video(center_crop_seq, frames, action, i, fps)
        provide_aug_update("Center crop", i)

        i += 1
        augment_video(flip_seq, aug_frames, action, i, fps)
        provide_aug_update(" + Flip", i)

    # Aug count: 49 (+12 x2)
    # Crop video from specific corner to specific dimensions
    corners = ['tl', 'tr', 'bl', 'br']
    for factor in factors:
        for corner in corners:
            corner_crop_seq =  va.Sequential([ va.CornerCrop(size=(int(height * factor), int(width * factor )), crop_position=corner) ])

            i+=1
            aug_frames = augment_video(corner_crop_seq, frames, action, i, fps)
            provide_aug_update("Corner crop", i)

            i += 1
            augment_video(flip_seq, aug_frames, action, i, fps)
            provide_aug_update(" + Flip", i)
    
    # # Aug count: 50 (+1)
    # temp_fit_seq = va.Sequential([ va.TemporalFit(size=48) ])
    
    # i+=1
    # augment_video(temp_fit_seq, frames, action, i, fps)
    # provide_aug_update("Temporal Fit", i)

    return i

def main():
    start_time = time.time()

    # Get all actions/gestures names
    actions = listdir(DATA_PATH)
    for action in actions:
        aug_count = 0
        print(f"\n--- Starting {action} augmentation ---")

        # Get all filenames for augmentating each
        videos = listdir(f"{DATA_PATH}/{action}")

        # Augment video by video
        for video in videos:
            print(f"\n-- Augmenting video {video} --")

            # Capture frames per video
            frames, fps = store_frames(f"{DATA_PATH}/{action}/{video}")

            # Augment video using frames
            aug_count = augmentation_sequences(aug_count, frames, action, fps)
        
    end_time = time.time()
    print("\nTotal Data Augmentation Time (s): ", end_time - start_time)

main()