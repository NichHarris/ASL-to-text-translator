
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



'''
# Detect and Predict Data Points in img using Holistic Model
def mp_detect_datapoints(img, model):
    # Convert img from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Improve Performance by Passing img by Reference through Not Writeable
    img.flags.writeable = False

    # Process img using Holistic Model
    res = model.process(img)

    # Make img Writeable Again to Reconvert to BGR
    img.flags.writeable = True

    # Convert img back to BGR from RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img, res

# Draw/Visualize Landmarks to Frame
def draw_landmarks(img, res):
    # Change Landmark (Blue Dot) and Connection (Red Line) Colors
    landmark_color = mp_drawing.DrawingSpec(color=(139, 0, 0), thickness=1, circle_radius=2)
    connection_color = mp_drawing.DrawingSpec(color=(0, 0, 139), thickness=1, circle_radius=1)

    # Pose Landmarks
    mp_drawing.draw_landmarks(img, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_color, connection_color)

    # Left Hand Landmarks
    mp_drawing.draw_landmarks(img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_color, connection_color)

    # Right Hand Landmarks
    mp_drawing.draw_landmarks(img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_color, connection_color)

# Combine Landmark Datapoints (DPs)
def combine_landmarks(results, has_visibility):
    # Concatenate All Landmarks Datapoints into Landmarks Array
    landmarks = []

    # Return a Zero Array if Res is Empty
    if not results:
        # Mark Len: Pose Has 33 Lines with 4 DPs, Left and Right Have 22 Lines with 3 DPs
        mark_len = 33 * 4 if has_visibility else 22 * 3
        return np.zeros(mark_len)

    # Add All Landmarks DPs to Array
    for res in results.landmark:
        # Add Each Frame to Array
        landmarks.append(res.x)
        landmarks.append(res.y)
        landmarks.append(res.z)

        # Only Pose Frame Includes Visibility
        if has_visibility:
            landmarks.append(res.visibility)
    
    return landmarks

def combine_datapoints(res):
    # Concatenate Collected Values into One Array
    pose = combine_landmarks(res.pose_landmarks, True)
    left_hand = combine_landmarks(res.left_hand_landmarks, False)
    right_hand = combine_landmarks(res.left_hand_landmarks, False)

    return np.concatenate([pose, left_hand, right_hand])

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

# Main Function
def main():
    # Instantiate Model with Specific Detection and Tracking Confidence
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    # Crop Dimensions
    x0, x1, y0, y1 = 170, 640, 0, 300

    # Asl Words
    asl_words = get_asl_dict()

    # Access Camera Using Open CV
    cap = cv2.VideoCapture(video)

    # Calculate time using frames
    time, fps = 0, cap.get(cv2.CAP_PROP_FPS)
    stop_capture = False

    for word, timestamp in asl_words.items():
        # Make directory for word
        frame_num = 0

        log_dir = os.path.join(data_path, word)
        has_collection = os.path.isdir(log_dir)
        if not has_collection:
            os.makedirs(os.path.join(data_path, word))
        
        print(f"Recording {word} ...")

        while(time < timestamp):
            # Capture and Read Frame
            _, frame = cap.read()

            # Extract Region of Interest (ROI)
            roi = frame[y0:y1, x0:x1]

            # Make Prediction/Detection of Datapoints Using Holistic Model
            img, res = mp_detect_datapoints(roi, holistic)

            # Draw Landmarks/Keypoints
            draw_landmarks(img, res)

            # Display Frame for 10ms
            cv2.imshow(f"{word}", img)

            # Save Datapoints in file
            if not has_collection:
                dp = combine_datapoints(res)
                file_path = os.path.join(data_path, word, str(frame_num))
                np.save(file_path, dp)
            
            print(len(combine_datapoints(res)))
            # Update time
            time += 1/fps
            frame_num += 1

            # Break if Press Q to Quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                stop_capture = True

        # Cleanup - Close Window
        cv2.destroyWindow(f"{word}")

        # Stop Collection Data
        if stop_capture:
            break
    
    # End Capture
    cap.release()
    cv2.destroyAllWindows()


print(cv2.rotate())
'''
