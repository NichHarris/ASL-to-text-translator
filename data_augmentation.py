'''
https://github.com/okankop/vidaug
Vidaug - video augmentation for deep learning
pip3 install vidaug

Data augmentation - alter existing training data to produce new artificial data
-> prevents overfitting 
-> increases training set size
-> improves model performance
-- Geometric transforms: crop, rotate and translate
-- Color space transforms, kernel filters and noise injection (Not used due to using hand and pose estimation over images)
-- Speed change, perspective skewing, elastic distortions, rotating, shearing, cropping, mirroring
https://neptune.ai/blog/data-augmentation-in-python
'''

# pi opencv-python scipy scikit-image numpy vidaug

import cv2
from vidaug import augmentors as va
import numpy as np
import time

start_time = time.time()

# Open sign language video file
video = "./data/bye/bye_0.mp4"
cap = cv2.VideoCapture(video)

# Calculate frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)

height, width, _ = frames[0].shape

# Define augmentation sequences
seq = va.Sequential([
    # Spatial shear in X, Y
    # va.RandomShear(x=0.05, y=0.05),
    # va.RandomShear(x=0.075, y=0.075),
    va.RandomShear(x=0.1, y=0.1),

    # # Translate in X, Y 
    # va.RandomTranslate(x=25, y=25),
    # va.RandomTranslate(x=75, y=75),
    # va.RandomTranslate(x=70, y=70),

    # va.HorizontalFlip(),

    # # Crop video from center and random corner
    # va.CenterCrop(size=(int(height / 1.2), int(width / 1.2))),
    # va.CenterCrop(size=(int(height / 1.4), int(width / 1.4))),
    # va.CornerCrop(size=(int(height / 1.1), int(width / 1.1))),
    # va.CornerCrop(size=(int(height / 1.15), int(width / 1.15)), crop_position='tl'),

    # # Rotate with degrees - stopped working...
    # va.RandomRotate(degrees=5),
    # va.RandomRotate(degrees=7),
    # va.RandomRotate(degrees=10),

    # Down and up sample - not working ...
    # va.Downsample(0.9)
    #va.Upsample(1.1)
])



# Aug count: 3 (+3)
# Spatial shear in X, Y
shear_seq = va.Sequential([ va.RandomShear(x=0.1, y=0.1) ])

# Aug count: 9 (+6)
# Translate in X, Y 
translate_seq = va.Sequential([ va.RandomTranslate(x=50, y=50) ])

# Aug count: 12 (+3)
# Crop video from center to specific dimensions
factors = [0.925, 0.95, 0.975]
for factor in factors:
    center_crop_seq =  va.Sequential([ va.CenterCrop(size=(int(height * factor), int(width * factor ))) ])
     
    # TODO: Apply on video here...

# Aug count: 24 (+12)
# Crop video from specific corner to specific dimensions
corners = ['tl', 'tr', 'bl', 'br']
for factor in factors:
    for corner in corners:
        corner_crop_seq =  va.Sequential([ va.CornerCrop(size=(int(height * factor), int(width * factor )), crop_position=corner) ])

        # TODO: Apply on video here...

# Aug count: 25 (+1) -> 50 (x2)
flip_seq = va.Sequential([ va.HorizontalFlip() ])

# Augment frames
aug_vid = flip_seq(seq(frames))
height, width, _ = aug_vid[0].shape

# Save augmented data
# Define codec and video writer (Four character code for uniquely identifying file formats)
fourcc = 'mp4v'
video_writer = cv2.VideoWriter_fourcc(*fourcc)

# Save video
out = cv2.VideoWriter("./test/bye_aug_3.mp4", video_writer, fps, (width, height))
for frame in aug_vid:
    out.write(frame)
out.release()

    

end_time = time.time()
print("Data Generation Time: ", end_time-start_time)