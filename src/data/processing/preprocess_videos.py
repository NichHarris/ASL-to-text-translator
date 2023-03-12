
'''
Mediapipe - provides ML solution to detect face landmarks, estimate human pose, and track hand position using its Holistic solution
- Regions of interest (ROI): x, y, z coordinates provided
    - x and y are normalized with respect to width and height of the frame
    - z represents depth of hand placement from camera
- Returns 543 landmarks (468 for face, 33 for pose, 21 per hand)
- Able to determine 3d positioning using multiple models on its ML pipeline
    - Palm detection model applied on entire image provides the bounds for each hand and crops the image to only contain the hand (95.7% precision)
    - Hand landmark model is then applied to determine the 3d hand landmarks (supervised training with 30k images of labeled data)
    - Use of palm detection reduces the amount of data required to perform hand keypoint detection by reducing the region and simplifying the task 
    therefore increases accuracy for determining coordinates
https://google.github.io/mediapipe/

Video to keypoints preprocessing
- Extract landmarks using mediapipe from augmented video dataset to produce training and testing data instances

- Each video stored as an array in a single file
    -> Each frame stored and converted to a 1d array of mediapipe keypoints 
    ...
- Execution Time: 3h 52m for 5 gestures

- Latest Execution: 520s for 10 gestures


Divide neural network scripts into folders
- ...
- ...
- ...

'''

# pip3.10 install matplotlib mediapipe torch torchvision

import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from os import listdir, makedirs, path  

DATA_PATH = "../../../inputs/raw"
# "../../../data_nick"
PREPROCESS_PATH = "../../../inputs/interim"

# Total landmarks: 21*3 * 2 (L/R Hands) + 25*3 (Pose)
NUM_LANDMARKS = 201 
HAND_LANDMARKS = 21 * 3
POSE_LANDMARKS = 25 * 3

# Process and convert video to landmarks array
def processing_data(cap, holistic):
    # Store processed frames 
    processed_frames = []

    # Initialize pose and left/right hand tensors
    left_hand, right_hand, pose = torch.zeros(HAND_LANDMARKS), torch.zeros(HAND_LANDMARKS), torch.zeros(POSE_LANDMARKS)

    while(True):
        # Capture and read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Pass frame to model by reference (not writeable) for improving performance
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image using Holistic model (Detect and predict keypoints)
        results = holistic.process(frame)

        # Ignore frames with no detection of both hands
        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            continue

        # No hand detected
        if not results.left_hand_landmarks:
            left_hand = torch.zeros(HAND_LANDMARKS)
        # Hand detected
        else:
            # Add left hand keypoints (21 w/ 3d coordinates)
            lh = results.left_hand_landmarks
            for i, landmark in enumerate(lh.landmark):
                shift_ind = i * 3
                left_hand[shift_ind] = landmark.x
                left_hand[shift_ind + 1] = landmark.y
                left_hand[shift_ind + 2] = landmark.z            
    
        if not results.right_hand_landmarks:
            right_hand = torch.zeros(HAND_LANDMARKS)
        else:
            # Add right hand keypoints (21 w/ 3d coordinates)
            rh = results.right_hand_landmarks
            for j, landmark in enumerate(rh.landmark):
                shift_ind = j * 3
                right_hand[shift_ind] = landmark.x
                right_hand[shift_ind + 1] = landmark.y
                right_hand[shift_ind + 2] = landmark.z

        # No pose detected
        if not results.pose_landmarks:
            pose = torch.zeros(POSE_LANDMARKS)
        # Pose detected
        else:
            # Add pose keypoints (25 w/ 3d coordinates)
            pl = results.pose_landmarks
            for k, landmark in enumerate(pl.landmark):
                # Ignore lower body (landmarks #25-33)
                if k >= 25:
                    break

                shift_ind = k * 3
                pose[shift_ind] = landmark.x
                pose[shift_ind + 1] = landmark.y
                pose[shift_ind + 2] = landmark.z  

        # Add processed frames, each as tensor
        processed_frames.append(torch.cat([left_hand, right_hand, pose], 0))

    return processed_frames

def main():
    start_time = time.time()
    if not path.exists(PREPROCESS_PATH):
        makedirs(PREPROCESS_PATH)

    # Get Mediapipe holistic solution
    mp_holistic = mp.solutions.holistic

    # Instantiate holistic model, specifying minimum detection and tracking confidence levels
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

    # Get all actions/gestures names
    actions = listdir(DATA_PATH)
    num_actions = len(actions)
    for i, action in enumerate(actions):
        print(f"\n--- Starting {action} preprocessing ({i + 1}/{num_actions}) ---")
        preprocess_folder = f"{PREPROCESS_PATH}/{action}"
        if not path.exists(preprocess_folder):
            makedirs(preprocess_folder)

        # Get all filenames for preprocessing each
        action_folder = f"{DATA_PATH}/{action}"
        videos = listdir(action_folder)

        # Preprocess video by video
        for video in videos:
            vid_name = video.split(".")[0]
            # Skip already processed videos
            # if path.exists(f'{preprocess_folder}/{vid_name}.pt'):
            #     continue

            print(f"\n-- Preprocessing video {vid_name} --")

            # Open sign language video file capture
            cap = cv2.VideoCapture(f"{action_folder}/{video}")

            # Processing video using frames
            processed_frames = processing_data(cap, holistic)
            print(f"Processed: {len(processed_frames)}")

            # Save processed data as torch file
            torch.save(processed_frames, f'{preprocess_folder}/{vid_name}.pt')

            # Close sign language video file capture
            cap.release()
        
    end_time = time.time()
    print("\nTotal Video to Landmarks Preprocessing Time (s): ", end_time - start_time)

main()