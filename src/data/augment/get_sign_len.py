import os
import torch

INTERIM_FOLDER='../../../inputs/interim'

avg_ds = 0
actions = os.listdir(INTERIM_FOLDER)
for action in actions:
    videos = os.listdir(f'./{INTERIM_FOLDER}/{action}')
    num_vids = len(videos)
    sum_frames = 0
    for video in videos:
        # # For custom dataset avg
        # if '_nick' not in video:
        #     num_vids -= 1
        #     continue

        # For WLASL dataset avg
        if '_nick' in video:
            num_vids -= 1
            continue

        frames = torch.load(f'./{INTERIM_FOLDER}/{action}/{video}')
        # print(f'Total for {video}: {len(frames)}')        
        sum_frames += len(frames)
    
    if num_vids == 0:
        avg_frames = 0
    else:
        avg_frames = sum_frames / num_vids
    print(f'{action}: {avg_frames}')
    avg_ds += avg_frames

print(avg_ds)

'''
WLASL Avg Frames per Sign:
easy: 44.90909090909091
please: 53.416666666666664
like: 47.95
happy: 49.92857142857143
why: 37.857142857142854
sad: 48.5
no: 50.77777777777778
she: 48.125
meet: 42.333333333333336
bad: 35.142857142857146
bye: 51.0
want: 37.8235294117647
more: 49.69230769230769
thank you: 53.18181818181818
good: 34.642857142857146
me: 41.125
hello: 46.7
you: 36.92857142857143
yes: 42.526315789473685
sorry: 50.27272727272727

Total Avg: 45.1416

----------------------------------

Custom Dataset Avg Frames per Sign:
easy: 35.22222222222222
please: 25.9
like: 40.4
happy: 31.7
why: 35.7
sad: 32.15
no: 28.65
she: 23.85
meet: 36.9
bad: 30.4
bye: 39.5
want: 41.35
more: 25.8
thank you: 38.15
good: 39.04761904761905
me: 25.35
hello: 24.55
you: 23.57894736842105
yes: 32.1
sorry: 30.4

Total Avg: 32.0349
'''
