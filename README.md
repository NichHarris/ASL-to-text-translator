# Intera - An American Sign Language/Speech to Text Conversation Mediator
COEN490 - Capstone Project

# Setup guide:

## Setting up virtual environment
Ensure you Python version is either 3.7 or 3.8 and 64-bit

### For macOS/linux:

```
pip3 install venv
python -m venv asl
source asl/bin/activate
pip3 install -r requirements.txt
```

### For Windows:

```
pip install venv
python -m venv asl
asl/Scripts/activate
pip install -r requirements.txt
```

# File Structure

inputs - Data
- custom: Recorded videos following the project requirements to supplement WLASL dataset
- raw: Original videos downloaded from WLASL dataset using video_downloader.py (to provide a variety of native signers)
- interim: Extracted landmarks from raw videos with mediapipe using preprocess_videos.py (to reduce neural network complexity and size)
- augmented: Augmented interim files with matrix transformations using matrix_augmentation.py (to artificially increase dataset size)
- dataset: Fitted and finalized keypoints using data_temporal_fit.py (to train and test neural network)

models - PyTorch Models
- asl_model_v{VERSION}.pth
- asl_optimizer_v{VERSION}.pth

runs - Tensorboard Graphs
- Training vs Validation Loss Entries

src - Python Code
- data:
    - augment: Data augmentation on videos and landmarks
    - collection: Fast custom data creation on live video
    - download: Download WLASL dataset for specified words
    - processing: Process video to keypoints and fit landmarks for model
- logs - Script outputs
- model
    - asl_model.py: PyTorch model architecture
    - dataset_loader.py: Load input/dataset for training/testing in batches
    - live_test.py: Test model on live or custom collected videos
    - train_model.py: Train and save model 
Additional scripts


# Code Explanation
## Retrieving WSASL dataset

***TODO: Add source: Added and modified code from https://github.com/dxli94/WLASL ***

- Install ffmpeg (to convert swf to mp4) and required python dependencies (to download from asl and youtube)
- Specify the words of interest to download by modififying new_words array
- Download videos from WLASL.json by running `python3.10 video_downloader.py`

## Processing dataset files into numpy arrays

- After the files are organized, run the command `python extract_videos.py`
- This will create the numpy files with the data points from all the dataset videos
- This will take several hours (improvements to be made to add multithreading to speed up)

## Data augmentation and preprocessing
- Perform video augmentation on videos in ./data by running `python3.10 video_augmentation.py`
- Translate video to keypoints per frame on videos in ./data to torch files in ./preprocess by running `python3.10 process_data.py`
- Perform keypoint matrix augmentation on torch tensor files in ./preprocess by running `python3.10 matrix_augmentation.py`
- Fit each extracted video keypoint file to 48 frames by randomly upsampling and downsampling video from ./preprocess to ./dataset by running `python3.10 data_temporal_fit.py`

...


TODO: 
- Add description for new files
- Update file to work with new structure
- Update requirements.txt
- Add custom testing folder 
- Add custom videos, train model and test with live script for top 20 words
