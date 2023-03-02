
import requests 
from bs4 import BeautifulSoup
import cv2
import os
import numpy as np 

DOWNLOAD_FOLDER = '../../../inputs/raw-2'

base_url = 'http://dai.cs.rutgers.edu/dai/s'
home_url = f'{base_url}/signbank'

def extract_endpoint_from_js(js_code: str) -> str :
    start_ep = js_code.find("'") + 1
    end_ep = js_code.find("'", start_ep)
    return js_code[start_ep:end_ep]

def extract_id_from_endpoint(endpoint: str) -> str :
    start_id = endpoint.find("id=") + 3
    end_id = endpoint.find("&", start_id)
    return endpoint[start_id:end_id]

# Make request
headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}

def get_all_words():
    res = requests.get(home_url, headers=headers) 

    with open('asllrp_words_links.txt', 'w') as f:
        # Check if home page contains words
        soup = BeautifulSoup(res.content, "html.parser")
        option_words = soup.findAll('option')

        for option_word in option_words:
            js_code = option_word.get('ondblclick')
            if js_code:
                # TODO: Modify how word is saved using parentheses over space
                # *sign_word, _ = option_word.text.split("(")
                # sign_word = sign_word.strip().lower()

                sign_word, num_occurences = option_word.text.strip().rsplit(" ", 1)
                sign_word = sign_word.lower()
            
                # Deal with same word appearing multiple times for slang
                word_endpoint = extract_endpoint_from_js(js_code)

                # Write to word and endpoint to file
                f.write(f'{sign_word}\t{word_endpoint}\n')
            else:
                # Add space between unrelated entries
                f.write('\n')


recognized_sign_words = [
    'bad', 
    'wave-goodbye', #bye 
    'easy', 
    'good', 
    'happy', '(1h)happy', 
    'hello', 
    'like', 
    'ix-1p', #me
    'meet', 
    'more', 
    'no', 
    'please/(1hr)enjoy', #please
    'sad', 
    'ix:i', #she
    'sorry', 
    'thank-you', 
    'want', 
    'why', 
    'yes', 
    'ix-2p' #you
]

def get_endpoints():
    sign_endpoints = {}
    with open('asllrp_words_links.txt', 'r') as f:
        for line in f.readlines():
            # Remove new line and ignore empty lines
            line = line.strip()
            if not line:
                continue 

            # TODO: Modify how to obtain word
            word, endpoint = line.split("\t")
            # if word in recognized_sign_words:
            if len(word) <= 1:
                # print(f'{word} {endpoint}')
                
                # Deal with same word appearing multiple times for slang
                if sign_endpoints.get(word):
                    sign_endpoints[word] = [endpoint, *sign_endpoints[word]]
                else:
                    sign_endpoints[word] = [endpoint]
            
    print(sign_endpoints)


# sign_endpoints = {'bad': ['occurrence?id_SignBankVariant=506'], 'easy': ['occurrence?id_SignBankVariant=1065'], 'good': ['occurrence?id_SignBankVariant=4693'], 'thank you': ['occurrence?id_SignBankVariant=4665'], 'happy': ['occurrence?id_SignBankVariant=1460', 'occurrence?id_SignBankVariant=93'], 'hello': ['occurrence?id_SignBankVariant=1486'], 'like': ['occurrence?id_SignBankVariant=1723'], 'meet': ['occurrence?id_SignBankVariant=1810'], 'more': ['occurrence?id_SignBankVariant=1852'], 'no': ['occurrence?id_SignBankVariant=27'], 'sad': ['occurrence?id_SignBankVariant=2423'], 'sorry': ['occurrence?id_SignBankVariant=2633'], 'want': ['occurrence?id_SignBankVariant=3012'], 'bye': ['occurrence?id_SignBankVariant=3028'], 'why': ['occurrence?id_SignBankVariant=3060', 'occurrence?id_SignBankVariant=126'], 'yes': ['occurrence?id_SignBankVariant=43', 'occurrence?id_SignBankVariant=3102'], 'me': ['occurrence?id_SignBankVariant=1612'], 'you': ['occurrence?id_SignBankVariant=1615'], 'she': ['occurrence?id_SignBankVariant=1620']}
def download_from_endpoints(sign_endpoints):
    for sign in sign_endpoints.keys():
        print(f'-- {sign} --')
        if not os.path.exists(f'{DOWNLOAD_FOLDER}/{sign}'):
            os.makedirs(f'{DOWNLOAD_FOLDER}/{sign}')
        
        word_endpoints = sign_endpoints[sign]
        for word_endpoint in word_endpoints:
            res = requests.get(f'{base_url}/{word_endpoint}', headers=headers) 

            # Check if link contains embedded video
            soup = BeautifulSoup(res.content, "html.parser")
            input_videos = soup.findAll('input')

            for input_vid in input_videos:
                input_val = input_vid.get('value')
                js_code = input_vid.get('onclick')
                
                # Condition 1:
                # Isolated sign (Data source T, 1+ videos, download only first)

                # Condition 2: 
                # Isolated sign (Data source D with 2 views in one video)

                # Note: Skipping videos from sentence due to being too short
                # Sign extracted from sentence (Data source F, very short)
                # Sign extracted from sentence (Data source S, very short with 2 views in one video)

                # Condition 3:
                # Isolated sign (Datasource R)
                if input_val != 'Original sign video' and (input_val != 'Sign video' or not js_code or not 'datasource=D' in js_code) and (not js_code or not 'datasource=R' in js_code):
                    continue
                
                word_video_endpoint = extract_endpoint_from_js(js_code)
                res = requests.get(f'{base_url}/{word_video_endpoint}', headers=headers) 

                video_id = extract_id_from_endpoint(word_video_endpoint)

                # Check if link contains embedded video
                soup = BeautifulSoup(res.content, "html.parser")
                videos = soup.findAll('source')
                
                data_source = word_video_endpoint[-1]
                video, *_ = videos
                video_url = video['src']
            
                # Download all videos - Split data source D in half
                file_name = f'{DOWNLOAD_FOLDER}/{sign}/{sign}_{video_id}_{data_source}.mp4'
                if data_source == 'D':
                    cap = cv2.VideoCapture(video_url)
                    vid_frames = []
                    
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    width = int(original_width / 2)
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    # Define codec and video writer (Four character code for uniquely identifying file formats)
                    fourcc = 'mp4v'
                    video_writer = cv2.VideoWriter_fourcc(*fourcc)

                    # Save video 
                    out = cv2.VideoWriter(file_name, video_writer, fps, (original_width, height))

                    # Define black bar regions
                    black_bar = int((original_width - width) / 2)
                    bars = np.zeros((height, black_bar, 3), dtype=np.uint8)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = frame[:,0:width,:]
                        out.write(np.hstack((bars, frame, bars)))
                        
                    out.release()
                    cap.release()
                else:
                    res = requests.get(video_url, headers=headers)
                    with open(file_name, 'wb') as file:
                        file.write(res.content)
                
                print(data_source, video_id, video_url)


# Script entry point
get_endpoints()