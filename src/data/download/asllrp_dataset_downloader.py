
import requests 
from bs4 import BeautifulSoup
import cv2
import os
import numpy as np 

DOWNLOAD_FOLDER = '../../../inputs/raw-2'

base_url = 'http://dai.cs.rutgers.edu/dai/s'
home_url = f'{base_url}/signbank'
sign_id_url = f'{base_url}/occurrence?id_SignBankVariant='

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
            if word in recognized_sign_words:
            # if recognized_sign_words in :
                # print(f'{word} {endpoint}')
                
                # Deal with same word appearing multiple times for slang
                if sign_endpoints.get(word):
                    sign_endpoints[word] = [endpoint, *sign_endpoints[word]]
                else:
                    sign_endpoints[word] = [endpoint]
            
    print(sign_endpoints)

sign_endpoints = {}
def download_from_endpoints():
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    for sign in sign_endpoints.keys():
        print(f'-- {sign} --')
        if not os.path.exists(f'{DOWNLOAD_FOLDER}/{sign}'):
            os.makedirs(f'{DOWNLOAD_FOLDER}/{sign}')
        
        word_endpoints = sign_endpoints[sign]
        for word_endpoint in word_endpoints:
            res = requests.get(f'{sign_id_url}{word_endpoint}', headers=headers) 

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


def get_english_translation(word, endpoint):
    # Get id from endpoint
    pre, id_no = endpoint.split('=')
    trans_endpoint = f'translation?id={id_no}'
    print(trans_endpoint)

    # Make get request
    trans_url = f'{base_url}/{trans_endpoint}'
    res = requests.get(trans_url, headers=headers) 

    # Extract translation using beautiful soup
    soup = BeautifulSoup(res.content, "html.parser")
    translation_html = soup.find('textarea', {'id': 'translation'})

    return translation_html.text if translation_html is not None else word

def save_english_word_links():
    with open('new_asl_words_links.txt', 'w') as new_file:
        with open('asllrp_words_links.txt', 'r') as asl_file:
            for line in asl_file:
                line = line.strip()
                if not line:
                    new_file.write("\n")
                    continue

                word, endpoint = line.split("\t")
                translation = get_english_translation(word, endpoint)
                print(translation)

                new_file.write(f"{translation}\t{endpoint}\n")

def get_dataset_words_endpoints():
    # 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    top_100 = ['afternoon', 'answer', 'big', 'buy', 'can', 'day']
    #  'evening', 'excuse', 'forget', 'give', 'hear', 'here', 'know', 'left', 'love', 'month', 'morning', 'name', 'night', 'out', 'question', 'read', 'remember', 'right', 'see', 'sell', 'small', 'take', 'think', 'time', 'today', 'tomorrow', 'understand', 'week', 'with', 'write', 'wrong', 'yesterday']
    # ['how', 'what', 'who', 'why', 'when', 'where', 'which']
    # {'afternoon', 'answer', 'bad', 'big', 'buy', 'bye', 'can', 'day', 'easy', 'evening', 'excuse me', 'forget', 'give', 'good', 'happy', 'hear', 'hello', 'here', 'in', 'know', 'left', 'like', 'love', 'me', 'meet', 'month', 'more', 'morning', 'name', 'night', 'no', 'out', 'please', 'question', 'read', 'remember', 'right', 'sad', 'see', 'sell', 'she', 'small', 'sorry', 'take', 'thank you', 'think', 'time', 'today', 'tomorrow', 'understand', 'want', 'week', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'write', 'wrong', 'yes', 'yesterday', 'you'}
    with open('last_38.txt', 'w') as top_100_file:
        with open('asl_words_en_translation_links.txt', 'r') as asl_file:
            i = 0
            for line in asl_file:
                # Remove new line and ignore empty lines
                line = line.strip()
                if not line:
                    continue

                # Check if the first word contains a substring of a word in the dictionary
                title, word, endpoint = line.split("\t")
                for top_one in top_100:
                    if top_one in word:
                        # Word found, sign gloss, english translation, endpoint
                        top_100_file.write(f"{top_one}\t{title}\t{word}\t{endpoint}\n")
                        break


# Script entry point
# get_endpoints()
# download_from_endpoints()
get_dataset_words_endpoints()