import cv2 
import mediapipe as mp
import torch
import random
import math
import os

from asl_model import AslNeuralNetwork, device

def processing_frame(frame, holistic):
    # Initialize pose and left/right hand tensoqs
    left_hand, right_hand, pose = torch.zeros(21 * 3), torch.zeros(21 * 3), torch.zeros(25 * 4)

    # Pass frame to model by reference (not writeable) for improving performance
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image using Holistic model (Detect and predict keypoints)
    results = holistic.process(frame)

    # Ignore frames with no detection of both hands
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return []

    # No hand detected
    if not results.left_hand_landmarks:
        left_hand = torch.zeros(21 * 3)
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
        right_hand = torch.zeros(21 * 3)
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
        pose = torch.zeros(25 * 4)
    # Pose detected
    else:
        # Add pose keypoints (25 w/ 3d coordinates plus visbility probability)
        pl = results.pose_landmarks
        for k, landmark in enumerate(pl.landmark):
            # Ignore lower body (landmarks #25-33)
            if k >= 25:
                break

            shift_ind = k * 4
            pose[shift_ind] = landmark.x
            pose[shift_ind + 1] = landmark.y
            pose[shift_ind + 2] = landmark.z  
            pose[shift_ind + 3] = landmark.visibility  

    # Concatenate processed frame
    return torch.cat([left_hand, right_hand, pose])

# Binary search on buffer frames (bf) using bit value (bv)
# Applicable to finding starting frame (bv = 1) and ending frame (bv = 0)
def bin_search(bf, bv, hm):
    # Search variables
    l, m, h = 0, 0, len(bf) - 1

    while l <= h:
        m = (h + l) // 2

        # Pass frame to mediapipe
        mv = processing_frame(bf[m], hm)

        # Found start frame with landmarks or end frame with no landmarks continue left
        if (bv == 1 and mv != []) or (bv == 0 and mv == []): 
            # Search left
            h = m - 1  
        else:
            # Search right
            l = m + 1

    return m

def get_holistic_model():
    # Get Mediapipe holistic solution
    mp_holistic = mp.solutions.holistic

    # Instantiate holistic model, specifying minimum detection and tracking confidence levels
    holistic = mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 
    
    return holistic

INPUT_SIZE = 226
NUM_SEQUENCES = 48
def live_video_temporal_fit(frames):
    # Calculate num frames over or under data frames input limit 
    num_frames = len(frames)
    missing_frames = NUM_SEQUENCES - num_frames

    if missing_frames == 0:
        print("Data already fitted to 48 frames")
        
    is_over_limit = missing_frames < 0
    print(f'Problem: {num_frames}')

    # Must select frames manually 
    missing_frames = abs(missing_frames)

    # Calculate num of times each frame and update frame population to properly sample missing frames
    frame_pop = range(num_frames)
    if num_frames == 0:
        # No frames in file
        print('Error - Empty video provided')
        return []

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
    torch_frames = torch.zeros([1, NUM_SEQUENCES, INPUT_SIZE], dtype=torch.float)
    for seq, frame in enumerate(frames):
        torch_frames[0][seq] = frame
    
    return torch_frames

cap = cv2.VideoCapture(0)
holistic = get_holistic_model()

frames = []
buffer_frames = []
 
END_SIGN_BUFFER = 3
SIGN_BUFFER_SIZE = 12

has_sign_started = False
is_sign_complete = False

# Collect frames until sign is complete
print('Video capture started...')
frame_count = 0
while not is_sign_complete:
    # Capture frame from camera
    ret, frame = cap.read()

    cv2.imshow('Frame', frame)
    cv2.waitKey(1)

    # Add frames to buffer
    buffer_frames.append(frame)
    if len(buffer_frames) >= SIGN_BUFFER_SIZE:
        frame_count += SIGN_BUFFER_SIZE
        print(f'Fc = {frame_count}')
        # Process every 12th frame (Once per second)
        processed_frame = processing_frame(frame, holistic)
        
        # Landmarks detected in frame
        if processed_frame != []:
            if not has_sign_started:
                # Set sign start flag
                has_sign_started = True
            
                # Find initial/first frame - delete all previous frames
                start_frame_ind = bin_search(buffer_frames, 1, holistic)
                buffer_frames = buffer_frames[start_frame_ind:]
                print(f'Start Video Check: {len(buffer_frames)}')

            # Add relevant start and middle frames
            frames.extend(buffer_frames)
        else:
            # Sign potentially completed if no landmarks detected anymore
            if has_sign_started:        
                # Find last/ending frame - delete all following frames
                end_frame_ind = bin_search(buffer_frames, 0, holistic)
                buffer_frames = buffer_frames[:end_frame_ind]    
                print(f'End Video Check: {len(buffer_frames)}')

                # Mark sign as complete if 3+ frames have no landmarks
                if SIGN_BUFFER_SIZE - len(buffer_frames) > END_SIGN_BUFFER:
                    is_sign_complete = True
                    has_sign_started = False

                # Add relevant end frames
                frames.extend(buffer_frames)
        
        # Remove useless frames, reset buffer
        buffer_frames = []

print('Video capture ended...')
print(f'Final Video: {len(frames)}')

# TODO: Add check to see if video is too short or too long

# Obtain fps (~29, expected 24)
fps = cap.get(cv2.CAP_PROP_FPS)
       
# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# Obtain width and height
height, width, _ = frames[0].shape

# Save augmented data
# Define codec and video writer (Four character code for uniquely identifying file formats)
fourcc = 'mp4v'
video_writer = cv2.VideoWriter_fourcc(*fourcc)

# Save video 
iteration = 1
sign_word = 'bye'
out = cv2.VideoWriter(f"{sign_word}_{iteration}.mp4", video_writer, fps, (width, height))
for frame in frames:
    out.write(frame)
out.release()



'''
# Fit
keypoints = live_video_temporal_fit(frames)
print(keypoints.size())

INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 3 # 3 LSTM Layers

LSTM_HIDDEN_SIZE = 128 # 128 nodes in LSTM hidden layers
FC_HIDDEN_SIZE = 64 # 64 nodes in Fc hidden layers
OUTPUT_SIZE = 5 # Starting with 5 classes = len(word_dict)

MODEL_PATH = "./model"
LOAD_MODEL_VERSION = "v1.0"
model_state_dict = torch.load(f'{MODEL_PATH}/asl_model_{LOAD_MODEL_VERSION}.pth')
model = AslNeuralNetwork(INPUT_SIZE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(model_state_dict)

VIDEOS_PATH = './data'
with torch.no_grad():
    y_pred = model(keypoints)
    print(y_pred)
    _, predicted = torch.max(y_pred.data, 1)

    signs = os.listdir(VIDEOS_PATH)
    print(signs[predicted])
'''