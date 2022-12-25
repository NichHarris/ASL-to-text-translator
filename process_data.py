
'''
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

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import torch
from matplotlib import pyplot as plt

# pi matplotlib torch torchvision mediapipe

# Get Mediapipe holistic solution
mp_holistic = mp.solutions.holistic

# Instantiate holistic model, specifying minimum detection and tracking confidence levels
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) 

# Open sign language video file
video = "./data/bye/bye_0.mp4"
cap = cv2.VideoCapture(video)

# Calculate number of frames
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize pose and left/right hand tensors
left_hand, right_hand, pose = torch.zeros(21, 3), torch.zeros(21, 3), torch.zeros(12, 4)

# processed_frames = torch.zeros(48, 54, 3)
processed_frames = []

frame_no = 1
while(frame_no < 200):
    # Capture and read frame
    ret, frame = cap.read()
    if not ret:
        break

    print(frame_no)

    # Pass frame to model by reference (not writeable) for improving performance
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image using Holistic model (Detect and predict keypoints)
    results = holistic.process(frame)

    frame_no += 1

    # Ignore frames with no detection of both hands
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        print("No result!!!!")
        continue

    # No hand detected
    if not results.left_hand_landmarks:
        print("No left hand detected...")
        left_hand = torch.zeros(21, 3)
    # Hand detected
    else:
        # Add hand keypoints (21 w/ 3d coordinates)
        lh = results.left_hand_landmarks
        for i, landmark in enumerate(lh.landmark):
            left_hand[i, 0] = landmark.x
            left_hand[i, 1] = landmark.y
            left_hand[i, 2] = landmark.z            
 
    if not results.right_hand_landmarks:
        print("No right hand detected...")
        right_hand = torch.zeros(21, 3)
    else:
        rh = results.right_hand_landmarks
        for i, landmark in enumerate(rh.landmark):
            right_hand[i, 0] = landmark.x
            right_hand[i, 1] = landmark.y
            right_hand[i, 2] = landmark.z

    # No pose detected
    if not results.pose_landmarks:
        print("No pose detected...")
        pose = torch.zeros(12, 4)
    # Pose detected
    else:
        # Add hand keypoints (21 w/ 3d coordinates)
        pl = results.pose_landmarks
        POSE_SHIFT = 11
        for i, landmark in enumerate(pl.landmark):
            # Ignore face mesh (landmarks #1-10) and lower body (landmarks #23-33)
            if i > 10 and i < 23:
                pose[i - POSE_SHIFT, 0] = landmark.x
                pose[i - POSE_SHIFT, 1] = landmark.y
                pose[i - POSE_SHIFT, 2] = landmark.z  
                pose[i - POSE_SHIFT, 3] = landmark.visibility  

    processed_frames.append({'left': left_hand, 'right': right_hand, 'pose': pose})

    print({'left': left_hand, 'right': right_hand, 'pose': pose })

print(processed_frames)
 
# Save processed data as torch file
# torch.save(processed_frames, f'bye_{i}.pt')

# End Capture
# cap.release()
# cv2.destroyAllWindows()


'''
# Create Folder to House Data Collections
def create_collection():
    # Action Detection - Using Sequence of Data Frames Over Single Frame
    # Array of Actions/Signs to Learn
    actions = np.array(["test", "test2", "nice"])

    # N Sequences/Videos for Each Action with N Frames/Snapshots Each
    num_frames = 20
    num_sequences = 20

    # Folder for Each Action/Sign
    for action in actions:
        # Folder for Each Sequence/Video
        for sequence in range(num_sequences):
            os.makedirs(os.path.join(data_path, action, f"{action}-{sequence}"))
'''
