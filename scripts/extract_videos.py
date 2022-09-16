# TODO:
# 1. Download video
# 2. Crop video - Find pixel range
# 3. Split video - Get only one word
# 4. Divide video into frames with cv2 on

import os
import time
import logging
import sys
import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt

SIGN_FILES_DIRECTORY = '../files/organized_files'
PROCESSED_FILES_DIRECTORY = '../files/processed_files'
LOGS = '../logs'

def init_log():
    if not os.path.exists(LOGS):
        os.mkdir(LOGS)
    logging.basicConfig(filename=f'{LOGS}/extract_videos.log', filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return 

# Get Holistic Model and Drawing Tools from Media Pipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Data Path Containing Collections
data_path = os.path.join("data")

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

def capture_vid(filename: str, index: int):
    cap = cv2.VideoCapture(filename)
    frame_num = 0

    # we want to capture 100 frames from each video
    # to do this, we need to limit or increase the rate of captures
    # if the video doesnt have 100 frames... well too bad
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames/fps
    print(f'Duration: {duration} FPS: {fps} Frames: {total_frames}')
    if total_frames < 100:
        logging.warn(f'Video: {filename} has less than 100 frames')


    if(cap.isOpened() == False):
        logging.warning(f'Unable to open or stream file {filename}, skipping...')
        return

    word = os.path.split(filename)[1].split('_')[0]
    
    while(cap.isOpened()):
        # Capture and Read Frame
        ret, frame = cap.read()

        # Extract Region of Interest (ROI)
        if not ret:
            break

        # Make Prediction/Detection of Datapoints Using Holistic Model
        img, res = mp_detect_datapoints(frame, holistic)

        # Draw Landmarks/Keypoints
        draw_landmarks(img, res)

        # Display Frame for 10ms
        cv2.imshow(f"{word}", img)

        # Save Datapoints in file
        if os.path.exists(f'{PROCESSED_FILES_DIRECTORY}/{word}'):
            dp = combine_datapoints(res)
            file_path = os.path.join(data_path, word, str(frame_num))
            np.save(f'{PROCESSED_FILES_DIRECTORY}/{word}/{index}/{word}_{index}_{frame_num}', dp)
        
        print(len(combine_datapoints(res)))
        frame_num += 1

        # Break if Press Q to Quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            stop_capture = True
    # Cleanup - Close Window
    cv2.destroyWindow(f"{word}")
    return

# Main Function
if __name__ == '__main__':
    init_log()

    # Instantiate Model with Specific Detection and Tracking Confidence
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    sign_list = os.listdir(SIGN_FILES_DIRECTORY)
    sign_list = sorted(sign_list)

    if not os.path.exists(PROCESSED_FILES_DIRECTORY):
        os.mkdir(PROCESSED_FILES_DIRECTORY)
    else:
        logging.info('Output directory already exists')

    for dir in sign_list:
        if os.path.isdir(f'{SIGN_FILES_DIRECTORY}/{dir}'):
            # continue inside this folder
            word_list = os.listdir(f'{SIGN_FILES_DIRECTORY}/{dir}')
            word_list = sorted(word_list)

            for index, sign in enumerate(word_list):
                if not os.path.exists(f'{PROCESSED_FILES_DIRECTORY}/{dir}'):
                    os.mkdir(f'{PROCESSED_FILES_DIRECTORY}/{dir}')

                # process video here
                if not os.path.exists(f'{PROCESSED_FILES_DIRECTORY}/{dir}/{index}'):
                    os.mkdir(f'{PROCESSED_FILES_DIRECTORY}/{dir}/{index}')
                capture_vid(f'{SIGN_FILES_DIRECTORY}/{dir}/{sign}', index)

        else:
            logging.info('File is not a directory, skipping...')

