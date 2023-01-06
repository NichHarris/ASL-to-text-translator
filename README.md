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

## Retrieving dataset

> Note: If you have access to the raw_videos_mp4.zip file skip this step....

***TODO: Add source: https://github.com/dxli94/WLASL ***

- Follow the direction for installing the required videos from this repository `https://github.com/dxli94/WLASL`
- A copy of script used for downloading, with some improvements will be added to this repository under `scripts`
- Recommendation is to use this script, as it organizes the files properly

## Organizing the dataset files

- Place zip/directory under the `files` directory
- Under scripts directory run the command `python organize_files.py <filename>` (if it is a zip, keep the extension)
> Note: if the directory is /raw_videos_mp4 you do not need to define a filename when running the command

## Processing the dataset files into numpy arrays

- After the files are organized, run the command `python extract_videos.py`
- This will create the numpy files with the data points from all the dataset videos
- This will take several hours (improvements to be made to add multithreading to speed up)

# ----- #

## Data augmentation
- Perform video augmentation on videos in ./data by running `python3.10 video_augmentation.py`
- Translate video to keypoints per frame on videos in ./data to torch files in ./preprocess by running `python3.10 process_data.py`
- Perform keypoint matrix augmentation on torch tensor files in ./preprocess by running `python3.10 matrix_augmentation.py`
- Fit each extracted video keypoint file to 48 frames by randomly upsampling and downsampling video from ./preprocess to ./dataset by running `python3.10 data_temporal_fit.py`


TODO: Continue

