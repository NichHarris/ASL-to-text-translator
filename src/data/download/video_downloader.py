import os
import sys
import json
import time
import random

import urllib.request
from pytube import YouTube
import cv2

from converter import Converter

# Modified code from WLASL Github repository
# https://github.com/dxli94/WLASL/blob/master/start_kit/video_downloader.py

DATASET_VIDEOS_INDEX_FILE='WLASL.json'
SAVE_VIDEOS_FOLDER='./input/raw'
# '../../../input/raw'

REGULAR_COLOR = '\033[0;0m'
FAILURE_COLOR = '\033[1;31m'
SUCCESS_COLOR = '\033[1;32m'

#pip3.10 install pytube opencv-python PythonVideoConverter

def request_video(url, referer=''):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}
    if referer:
        headers['Referer'] = referer

    print(f'Requesting {url}')
    req = urllib.request.Request(url, None, headers) 
    res = urllib.request.urlopen(req)
    
    data = res.read()
    return data


def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

def save_video_mp4(vid_frames, vid_dest, vid_fps, vid_dim):
    # Define codec and video writer (Four character code for uniquely identifying file formats)
    fourcc = 'mp4v'
    video_writer = cv2.VideoWriter_fourcc(*fourcc)

    # Save video 
    out = cv2.VideoWriter(vid_dest, video_writer, vid_fps, vid_dim)
    for frame in vid_frames:
        out.write(frame)
    out.release()


def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, f'{video_id}.swf')
    if os.path.exists(saveto):
        print(f'{video_id} exists at {save_to}')
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)

    # Convert swf to mp4
    convert_swf_mp4_cmd = f'ffmpeg -loglevel panic -i {saveto} -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" {dirname}/{video_id}.mp4'
    os.system(convert_swf_mp4_cmd)
    os.remove(saveto)

def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, f'{video_id}.mp4')
    if os.path.exists(saveto):
        print(f'{video_id} exists at {saveto}')
        return 
    
    data = request_video(url)
    save_video(data, saveto)

def download_youtube(url, dirname, video_id, start_frame, end_frame):
    vid_file = os.path.join(dirname, f'{video_id}.mp4')
    if os.path.exists(vid_file):
        print(f'{video_id} exists at {dirname}')
        return 
    
    print(f'Requesting {url}')
    yt = YouTube(url)
    mp4_streams = yt.streams.filter(file_extension='mp4')

    yt_stream = mp4_streams.first()
    yt_stream.download(output_path=dirname, filename=f'{video_id}.mp4')

    # Extract specific frame range
    if start_frame != 1 or end_frame != -1:
        print(f'Download part of video: {start_frame}, {end_frame}')

        cap = cv2.VideoCapture(vid_file)
        count = 1
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if start_frame <= count and (end_frame >= count or end_frame == -1):
                frames.append(frame)
            
            count += 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width, _ = frames[0].shape

        cap.release()
        
        os.remove(vid_file)
        save_video_mp4(frames, vid_file, fps, (width, height))

def download_videos(indexfile, saveto):
    print('--- Started download ---')
    json_file = open(indexfile)
    content = json.load(json_file)
    
    sign_words = ['hello', 'bye', 'me/I', 'you', 'good', 'yes', 'no', 'thank you', 'please', 'he/she/they']
    new_sign_words = ['hello'] #['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    # Note: c is missing in words

    if not os.path.exists(saveto):
        os.mkdir(saveto)
    
    for entry in content:
        gloss = entry['gloss']
        if gloss in new_sign_words:
            new_saveto = f'{saveto}/{gloss}'
            if not os.path.exists(new_saveto):
                os.makedirs(new_saveto)

            new_saveto = f'{saveto}/{gloss}'
            print(f'-- gloss: {gloss} --')

            instances = entry['instances']
            for inst in instances:
                video_url = inst['url']
                video_id = gloss + "_" + inst['video_id']
                start_frame = inst['frame_start']
                end_frame = inst['frame_end']
                print(f'video: {video_id}')

                try:
                    if 'aslpro' in video_url:
                        download_aslpro(video_url, new_saveto, video_id)
                    elif 'youtube' in video_url or 'youtu.be' in video_url:
                        download_youtube(video_url, new_saveto, video_id, start_frame, end_frame)
                    else:
                        download_others(video_url, new_saveto, video_id)

                    sys.stdout.write(SUCCESS_COLOR)
                    print(f'Successful download - video {video_id}!\n')
                    sys.stdout.write(REGULAR_COLOR)
                except Exception as e:
                    sys.stdout.write(FAILURE_COLOR)
                    print(f'Failed download - {str(e)} for video {video_id}\n')
                    sys.stdout.write(REGULAR_COLOR)
                
                # Pause between video download requests
                time.sleep(random.uniform(0.001, 0.01))

    print('--- Finished download ---')

if __name__ == '__main__':
    download_videos(DATASET_VIDEOS_INDEX_FILE, SAVE_VIDEOS_FOLDER)

