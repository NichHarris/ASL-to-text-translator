import cv2 
import mediapipe as mp
import torch
import random
import math
import os

from asl_model import AslNeuralNetwork

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
        
    is_over_limit = missing_frames < 0
    # print(f'Problem: {num_frames}')

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

def softmax(output):
    e = torch.exp(output)
    return e / e.sum()

INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 3 # 3 LSTM Layers

LSTM_HIDDEN_SIZE = 128 # 128 nodes in LSTM hidden layers
FC_HIDDEN_SIZE = 64 # 64 nodes in Fc hidden layers
OUTPUT_SIZE = 20 # Starting with 5 classes = len(word_dict)

MODEL_PATH = "../../models"
LOAD_MODEL_VERSION = "v2.9_20"
# v2.5_14 works great (82% and 31%)
# v2.9_15 works even better (88% and 50%)
def load_model():
    model = AslNeuralNetwork(INPUT_SIZE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, OUTPUT_SIZE)
    
    model_state_dict = torch.load(f'{MODEL_PATH}/asl_model_{LOAD_MODEL_VERSION}.pth')
    model.load_state_dict(model_state_dict)

    return model
    

VIDEOS_PATH = '../../inputs/interim'
# signs = sorted(os.listdir(VIDEOS_PATH))
signs = ['bad', 'bye', 'easy', 'good', 'happy', 'hello', 'like', 'me', 'meet', 'more', 'no', 'please', 'sad', 'she', 'sorry', 'thank you', 'want', 'why', 'yes', 'you']
def testing_data(sign_word, keypoints): 
    with torch.no_grad():
        y_pred = model(keypoints)
        _, predicted = torch.max(y_pred.data, 1)

        torch.set_printoptions(precision=5, sci_mode=False)
        print(sign_word, softmax(y_pred), signs[predicted])  
    
    return sign_word == signs[predicted]

# Live testing with opencv video camera
def live_single_sign_test():
    cap = cv2.VideoCapture(0)

    frames = []
    buffer_frames = []
    
    END_SIGN_BUFFER = 3
    SIGN_BUFFER_SIZE = 12

    has_sign_started = False
    is_sign_complete = False

    # Collect frames until sign is complete
    print('Video capture started...')
    frame_count = 0
    sign_word = input('Enter the word for the performed sign: ')
    while not is_sign_complete:
        # Capture frame from camera
        ret, frame = cap.read()

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

        # Add frames to buffer
        buffer_frames.append(frame)
        if len(buffer_frames) >= SIGN_BUFFER_SIZE:
            frame_count += SIGN_BUFFER_SIZE

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
                    print(f'Started sign video...')

                # Add relevant start and middle frames
                frames.extend(buffer_frames)
            else:
                # Sign potentially completed if no landmarks detected anymore
                if has_sign_started:        
                    # Find last/ending frame - delete all following frames
                    end_frame_ind = bin_search(buffer_frames, 0, holistic)
                    buffer_frames = buffer_frames[:end_frame_ind]    

                    # Mark sign as complete if 3+ frames have no landmarks
                    if SIGN_BUFFER_SIZE - len(buffer_frames) > END_SIGN_BUFFER:
                        is_sign_complete = True
                        has_sign_started = False
                        print(f'Ended sign video...')

                    # Add relevant end frames
                    frames.extend(buffer_frames)
            
            # Remove useless frames, reset buffer
            buffer_frames = []

    # TODO: Add check to see if video is too short (<12) or too long (>96) on len(frames)
    print(len(frames))

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Mediapipe keypoint extraction
    mp_frames = []
    for frame in frames:
        pcf = processing_frame(frame, holistic)
        if pcf != []:
            mp_frames.append(pcf)

    # Fit
    keypoints = live_video_temporal_fit(mp_frames)

    # Pass keypoints to model
    successful_pred = testing_data(sign_word, keypoints)
    if successful_pred:
        print('Succesfully predicted word live!')
    else:
        print('Please try again...')


# Automated testing from saved sample videos
def automated_testing():
    live_data_dir = '../../test_nick' #'../../test_ali' #
    video_names = os.listdir(live_data_dir)
    accuracy = 0
    for vidname in video_names:
        video = f'{live_data_dir}/{vidname}'
        sign_word = vidname.split('_')[0]

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

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

        # Mediapipe keypoint extraction
        mp_frames = []
        for frame in frames:
            pcf = processing_frame(frame, holistic)
            if pcf != []:
                mp_frames.append(pcf)

        # Fit
        keypoints = live_video_temporal_fit(mp_frames)

        torch.save(keypoints, f'../../processed_tests/nick/{vidname.split(".mp4")[0]}.pt')

        # Pass keypoints to model
        successful_pred = testing_data(sign_word, keypoints)
        if successful_pred:
            accuracy += 1
        
    print(signs)
    print(f'{accuracy}/{len(video_names)} = {accuracy/len(video_names)}')
    print(f'using model {LOAD_MODEL_VERSION}')

def fast_automated_testing():
    accuracy = 0
    processed_testing_path = f'../../processed_tests/ali'

    videos = os.listdir(processed_testing_path)
    for video in videos:
        sign_word = video.split('_')[0]
        keypoints = torch.load(f'{processed_testing_path}/{video}')

        # Pass keypoints to model
        successful_pred = testing_data(sign_word, keypoints)
        if successful_pred:
            accuracy += 1
    
    print(signs)
    print(f'{accuracy}/{len(videos)} = {accuracy/len(videos)}')
    print(f'using model {LOAD_MODEL_VERSION}')


holistic = get_holistic_model()

# Notes:
# Model 1.0, 1.2, 1.3 (1.1 with dataset3)
# Model 2.0 provides 87% on my dataset
# Model 2.5 
# Model 2.9 v20 85% and 51%
model = load_model()

is_automated = True
is_fast_comparison = True
if is_automated:
    if is_fast_comparison:
        fast_automated_testing()
    else:
        automated_testing()
else:
    live_single_sign_test()