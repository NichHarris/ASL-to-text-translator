import os
import logging
import zipfile
import sys

LOGS = '../logs'
FILES_DIRECTORY = '../files'
VIDEO_FILE_NAME = 'raw_videos_mp4'
OUTPUT_DIR = '../files/organized_files'
COMPOUND_VIDEO_DIR = '../files/extra_videos'
MAX_SIZE = 2000

def init_log():
    if not os.path.exists(LOGS):
        os.mkdir(LOGS)
    logging.basicConfig(filename=f'{LOGS}/organize_files.log', filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return 

def retrieve_files(filename: str):
    if os.path.exists(OUTPUT_DIR):
        logging.info('Organized files directory already exsits...')
        return

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
    add_path = OUTPUT_DIR
    video_count = 0
    dir_count = 0

    try:
        os.mkdir(OUTPUT_DIR)
        os.mkdir(COMPOUND_VIDEO_DIR)
    except Exception as e:
        logging.error(f'ERROR -Unable to create directory {OUTPUT_DIR}: {str(e)}')
        return

    for video_count, video in enumerate(video_list):
        video_name = os.path.split(video)[1]
        if not video_name:
            logging.warning('Invalid file name, skipping')
            continue
        
        video_name = video_name.split('_')[0]
        
        # make new dir for new word
        if video_name != last_word:
            add_path = f'{OUTPUT_DIR}/{video_name}'
            word_occurence_count = 0
            last_word = video_name
            dir_count += 1
            os.mkdir(add_path)
        
        # add it to the directory
        size = os.stat(f'{FILES_DIRECTORY}/{filename}/{video}').st_size/1024
        if size < MAX_SIZE:
            os.rename(f'{FILES_DIRECTORY}/{filename}/{video}', f'{add_path}/{video_name}_{word_occurence_count}.mp4')
        else:
            os.rename(f'{FILES_DIRECTORY}/{filename}/{video}', f'{COMPOUND_VIDEO_DIR}/{video_name}_{word_occurence_count}.mp4')
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

