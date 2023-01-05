
'''
https://google.github.io/mediapipe/
Mediapipe - provides ML solution to detect face landmarks, estimate human pose, and track hand position using its Holistic solution
-> Regions of interest (ROI)
-> Returns 543 landmarks (468 for face, 33 for pose, 21 per hand)
-> Able to determine 3d positioning using multiple models on its ML pipeline
-- Palm detection model applied on entire image provides the bounds for each hand and crops the image to only contain the hand (95.7% precision)
-- Hand landmark model is then applied to determine the 3d hand landmarks (supervised training with 30k images of labeled data)
-- Use of palm detection reduces the amount of data required to perform hand keypoint detection by reducing the region and simplifying the task 
therefore increases accuracy for determining coordinates

- x, y, z coordinates provided
-> X and y are normalized with respect to width and ehight of the frame
-> Z represents depth of hand placement from camera
'''

# pip3.10 install matplotlib mediapipe torch torchvision

import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from os import listdir, makedirs, path

DATA_PATH = "./data"
PREPROCESS_PATH = "./preprocess"
NUM_LANDMARKS = 174 # 21*3 * 2 + 12 * 4

def processing_data(cap, holistic):
    # Initialize pose and left/right hand tensors
    left_hand, right_hand, pose = torch.zeros(21 * 3), torch.zeros(21 * 3), torch.zeros(12 * 4)

    processed_frames = []

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
            #print(f"Skipping frames: No hands detected!")
            continue

        # No hand detected
        if not results.left_hand_landmarks:
            #print("No left hand detected...")
            left_hand = torch.zeros(21 * 3)
        # Hand detected
        else:
            # Add hand keypoints (21 w/ 3d coordinates)
            lh = results.left_hand_landmarks
            for i, landmark in enumerate(lh.landmark):
                shift_ind = i * 3
                left_hand[shift_ind] = landmark.x
                left_hand[shift_ind + 1] = landmark.y
                left_hand[shift_ind + 2] = landmark.z            
    
        if not results.right_hand_landmarks:
            #print("No right hand detected...")
            right_hand = torch.zeros(21 * 3)
        else:
            rh = results.right_hand_landmarks
            for j, landmark in enumerate(rh.landmark):
                shift_ind = j * 3
                right_hand[shift_ind] = landmark.x
                right_hand[shift_ind + 1] = landmark.y
                right_hand[shift_ind + 2] = landmark.z

        # No pose detected
        if not results.pose_landmarks:
            #print("No pose detected...")
            pose = torch.zeros(12 * 4)
        # Pose detected
        else:
            # Add hand keypoints (21 w/ 3d coordinates)
            pl = results.pose_landmarks
            POSE_SHIFT = 11
            for k, landmark in enumerate(pl.landmark):
                # Ignore face mesh (landmarks #1-10) and lower body (landmarks #23-33)
                if k > 10 and k < 23:
                    shift_ind = (k - POSE_SHIFT) * 4
                    pose[shift_ind] = landmark.x
                    pose[shift_ind + 1] = landmark.y
                    pose[shift_ind + 2] = landmark.z  
                    pose[shift_ind + 3] = landmark.visibility  

        # Add processed frames, each as tensor
        processed_frames.append(torch.cat([left_hand, right_hand, pose], 0))

    # Add frame count to processed frames
    # processed_frames.insert(0, {'no_frames': len(processed_frames)})
    return processed_frames

def main():
    start_time = time.time()
    if not path.exists(PREPROCESS_PATH):
        makedirs(PREPROCESS_PATH)

    # Get Mediapipe holistic solution
    mp_holistic = mp.solutions.holistic

    # Instantiate holistic model, specifying minimum detection and tracking confidence levels
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

    # Get all actions/gestures names
    actions = listdir(DATA_PATH)
    for action in actions:
        print(f"\n--- Starting {action} preprocessing ---")
        preprocess_folder = f"{PREPROCESS_PATH}/{action}"
        if not path.exists(preprocess_folder):
            makedirs(preprocess_folder)

        # Get all filenames for preprocessing each
        action_folder = f"{DATA_PATH}/{action}"
        videos = listdir(action_folder)

        # Preprocess video by video
        for video in videos:
            vid_name = video.split(".")[0]
            # Open sign language video file capture
            print(f"\n-- Preprocessing video {vid_name} --")
            cap = cv2.VideoCapture(f"{action_folder}/{video}")

            # Processing video using frames
            processed_frames = processing_data(cap, holistic)
            print(f"Processed: {len(processed_frames) - 1}")

            # Save processed data as torch file
            torch.save(processed_frames, f'{preprocess_folder}/{vid_name}.pt')

            # Close sign language video file capture
            cap.release()
        
    end_time = time.time()
    print("\nTotal Preprocessing Time (s): ", end_time - start_time)

main()
