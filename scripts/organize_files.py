import os
import logging
import zipfile
import sys
import cv2

LOGS = '../logs'
FILES_DIRECTORY = '../files'
VIDEO_FILE_NAME = 'raw_videos_mp4'
OUTPUT_DIR = '../files/organized_files'
COMPOUND_VIDEO_DIR = '../files/extra_videos'

def init_log():
    if not os.path.exists(LOGS):
        os.mkdir(LOGS)
    logging.basicConfig(filename=f'{LOGS}/organize_files.log', filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return 

def retrieve_files(filename: str):
    if os.path.exists(OUTPUT_DIR):
        logging.info('Organized files directory already exsits...')
    else:
        os.mkdir(OUTPUT_DIR)

    if zipfile.is_zipfile(filename):
        logging.info('File is a zip... unpacking\nThis may take a while...')
        with zipfile.ZipFile(filename, 'r') as zip:
            zip.extractall(f'{FILES_DIRECTORY}')

    try:
        os.remove(filename)
    except Exception as e:
        logging.debug(str(e))
    filename = os.path.splitext(filename)[0]
    
    if os.path.exists(filename) and os.path.isdir(filename):
        logging.info('Video file directory found, begining processing')
        organize_videos(filename)
    else:
        logging.error('Unable to find directory')
    return

def organize_videos(filename: str):
    video_list = os.listdir(filename)
    video_list = sorted(video_list)
    last_word = ''
    word_occurence_count = 0
    create_word_dir = OUTPUT_DIR
    video_count = 0
    dir_count = 0

    if os.path.exists(COMPOUND_VIDEO_DIR):
        logging.info(f'Unable to create directory {COMPOUND_VIDEO_DIR}')
    else:
        os.mkdir(COMPOUND_VIDEO_DIR)

    for video_count, video in enumerate(video_list):
        base_video = f'{filename}/{video}'
        video_name = os.path.split(video)[1]
        if not video_name:
            logging.warning('Invalid file name, skipping')
            continue
        
        video_name = video_name.split('_')[0]
        
        # make new dir for new word
        if video_name != last_word:
            create_word_dir = f'{OUTPUT_DIR}/{video_name}'
            word_occurence_count = 0
            last_word = video_name
            dir_count += 1
            if not os.path.exists(create_word_dir):
                os.mkdir(create_word_dir)
        
        # add it to the directory
        cap = cv2.VideoCapture(base_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames/fps

        # if longer than 10 seconds, assume it is a compound video
        if duration < 10:
            os.rename(base_video, f'{create_word_dir}/{video_name}_{word_occurence_count}.mp4')
        else:
            os.rename(base_video, f'{COMPOUND_VIDEO_DIR}/{video_name}_{word_occurence_count}.mp4')
        word_occurence_count += 1
    
    logging.info(f'Processing finished...\nVideos processed: {video_count}\nUnique words: {dir_count}')
    os.rmdir(filename)
    return


if __name__ == '__main__':
    init_log()
    logging.info('Running file organization script...')
    for i, arg in enumerate(sys.argv):
        if i == 1:
            VIDEO_FILE_NAME = str(arg)

    retrieve_files(f'{FILES_DIRECTORY}/{VIDEO_FILE_NAME}')    

