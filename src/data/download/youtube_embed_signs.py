import json
import requests
from bs4 import BeautifulSoup

# Write embedded links and word pairs to file
with open('./embedded_links.txt', 'w') as f:
    # Read json file
    content = json.load(open('WLASL.json'))
    for entry in content:
        gloss = entry['gloss']
        
        # Iterate through every instance 
        instances = entry['instances']
        for inst in instances:
            vid_url = inst['url']
            start_frame = inst['frame_start']
            end_frame = inst['frame_end']
            
            # Must be youtube link and only word signed in video
            if ('youtube' in vid_url or 'youtu.be' in vid_url) and (start_frame == 1 and end_frame == -1):
                embedded_id = vid_url[-11:]
                embedded_url = f'https://www.youtube.com/embed/{embedded_id}'

                # Make request
                headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}
                res = requests.get(embedded_url, headers=headers) 

                # Check if link contains embedded video
                soup = BeautifulSoup(res.content, "html.parser")
                video_embedded = soup.findAll('a')
                if len(video_embedded) != 0:
                    f.write(f'{gloss} {embedded_url}\n')
                    break