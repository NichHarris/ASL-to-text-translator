'''
Video: Temporal information (Sequence of frames, Sequential data)
- Training with image sequence (ie video) 
- Standard video frame rate is 24fps

--- Sequence Model --- 
RNN - Recurrent Neural Network (for processing sequential data)
-> Use Many to One Architecture (Multiple frames classified to single word)
- Slow to train
    -> Solved by Transformers
- Vanishing gradient problem
    -> Solved with Attention
- Long-term dependencies problem
    -> Solved by Long Short Term Memory (LSTM)
        - Internal memory improved for extended times by modifying hidden layer
        - Using forget (remove dependencies), input and output gates (add dependencies)

Normalisation - ...


Training and Validation Loss
https://www.baeldung.com/cs/training-validation-loss-deep-learning

- Validation Loss >> Training Loss: Overfitting
- Validation Loss > Training Loss: Some overfitting
    -> Val decrease then increase

- Validation Loss < Training Loss: Some underfitting 
- Validation Loss << Training Loss: Underfitting

- Validation Loss == Training Loss: No overfit or underfit
    -> Both decrease and stabilize
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

from torch.utils.data import DataLoader

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define hyper parameters
INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 3 # 3 LSTM Layers

# TODO: Hyperparam Optimization: 3-6 layers and 64-256 for lstm + 2-4 layers and 32-128 for fc
LSTM_HIDDEN_SIZE = 128 # 128 nodes in LSTM hidden layers
FC_HIDDEN_SIZE = 64 # 64 nodes in Fc hidden layers
OUTPUT_SIZE = 20 # Starting with 5 classes = len(word_dict) # TODO: Get from word_dict size

# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
# Optimal learning rate: Bt 0.0001 and 0.01 (Default: 0.001)
NUM_EPOCHS = 30
BATCH_SIZE = 256 # DO 512 next
LEARNING_RATE = 0.001

MODEL_PATH = "../../models"
LOAD_MODEL_VERSION = "v2.5"
NEW_MODEL_VERSION = "v2.9"

# Create model
model = AslNeuralNetwork(INPUT_SIZE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# -- Load Model -- #
# model_state_dict = torch.load(f'{MODEL_PATH}/asl_model_{LOAD_MODEL_VERSION}.pth')
# optimizer_state_dict = torch.load(f'{MODEL_PATH}/asl_optimizer_{LOAD_MODEL_VERSION}.pth')
# model.load_state_dict(model_state_dict)
# optimizer.load_state_dict(optimizer_state_dict)

# -- Constants -- #
VIDEOS_PATH = '../../inputs/interim'
DATASET_PATH = '../../inputs/dataset_only'
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)

TRAIN_SPLIT = int(DATASET_SIZE * 0.9)
TEST_SPLIT = DATASET_SIZE - TRAIN_SPLIT #(DATASET_SIZE * 0.1)

ACTUAL_TRAIN_SPLIT = int(TRAIN_SPLIT * 0.8)
VALID_SPLIT = TRAIN_SPLIT - ACTUAL_TRAIN_SPLIT #(TRAIN_SPLIT * 0.2)

# Define signed words/action classes 
word_dict = {'bad': 0, 'bye': 1, 'easy': 2, 'good': 3, 'happy': 4, 'hello': 5, 'like': 6, 'me': 7, 'meet': 8, 'more': 9, 'no': 10, 'please': 11, 'sad': 12, 'she': 13, 'sorry': 14, 'thank you': 15, 'want': 16, 'why': 17, 'yes': 18, 'you': 19}
# for i, sign in enumerate(sorted(os.listdir(VIDEOS_PATH))):
#     word_dict[sign] = i
# print(word_dict)

# -- Load dataset -- #
dataset = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)

# Split into training, validation and testing sets
train_split, test_split = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT])
train_split, valid_split = torch.utils.data.random_split(train_split, [ACTUAL_TRAIN_SPLIT, VALID_SPLIT])

# Define train, valid, test data loaders
train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_split, batch_size=BATCH_SIZE, shuffle=True)

train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = 30
no_valid_loss_items = 10

# Create summary writer for Tensorboard
writer = SummaryWriter()

# -- Train Model -- #
best_train_loss = 1
for epoch in range(NUM_EPOCHS):
    print(f'--- Starting Epoch #{epoch} ---')

    # -- Actual Training -- #
    mini_batch_loss = 0
    train_total_loss = 0
    for i, (keypoints, labels) in enumerate(train_loader):
        # Use GPU for model training computations  
        keypoints = keypoints.to(model.device)
        labels = labels.to(model.device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, labels)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training loss
        train_total_loss += loss.item()
        mini_batch_loss += loss.item()
        if (i + 1) % no_train_loss_items == 0:
            print(f"Training Loss - {i + 1}/{train_size}: {mini_batch_loss/no_train_loss_items}")
            mini_batch_loss = 0

    # -- Validation -- #
    mini_batch_loss = 0
    valid_total_loss = 0
    for i, (keypoints, labels) in enumerate(valid_loader):  
        # Use GPU for model validation computations  
        keypoints = keypoints.to(model.device)
        labels = labels.to(model.device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, labels)
        
        # Compute validation loss
        valid_total_loss += loss.item()
        mini_batch_loss += loss.item()
        if (i + 1) % no_valid_loss_items == 0:
            print(f'Validation Loss - {i + 1}/{valid_size}: {mini_batch_loss/no_valid_loss_items}')
            mini_batch_loss = 0
    
    # TODO: Implement early stopping with best validation loss and patience
    # Save model with early stopping to prevent overfitting
    if train_total_loss/train_size < best_train_loss:
        best_train_loss = train_total_loss/train_size
        torch.save(model.state_dict(), f'{MODEL_PATH}/asl_model_{NEW_MODEL_VERSION}.pth')
        print(f'Model Saved - Valid={valid_total_loss/valid_size} \t Train={train_total_loss/train_size}')
    
    # Track training and validation loss in Tensorboard
    writer.add_scalars(f'Loss Plot - nn_{NEW_MODEL_VERSION}', 
        {
            'train_loss': train_total_loss/train_size,
            'valid_loss': valid_total_loss/valid_size,
        }, 
        epoch 
    )

# View tensorboard using following command:
#tensorboard --logdir runs
writer.close()

# -- Test Model -- #
# Gradient computation not needed in testing phase
with torch.no_grad():
    # TODO; Convert to f1-measure instead of accuracy
    true_count = 0
    test_count = 0
    for keypoints, labels in test_loader:
        # Use GPU for model testing computations  
        keypoints = keypoints.to(model.device)
        labels = labels.to(model.device)
        
        # Pass testing instances
        output = model(keypoints)

        # Obtain max prob value and class index
        _, predicted = torch.max(output.data, 1)
        
        # TODO: Perform f1 measure instead
        true_count += (predicted == labels).sum().item()
        test_count += labels.size(0)
    
    model_accuracy = (true_count / test_count) * 100
    print(f'Model Accuracy: {model_accuracy} %')

    # TODO: Calculate and display confusion matrix
    # cm = confusion_matrix(labels, predicted)
    # sns.heatmap(cm, annot=True)

# -- Save Model -- #
torch.save(model.state_dict(), f'{MODEL_PATH}/asl_model_{NEW_MODEL_VERSION}.pth') 
torch.save(optimizer.state_dict(), f'{MODEL_PATH}/asl_optimizer_{NEW_MODEL_VERSION}.pth')


'''
Models
on 5 words
- 1.0 : Video augmented
- 1.1 : Matrix augmented with simple rotation 
- 1.2 : Matrix augmented with simple translation and complex rotation 
- 1.3 : Matrix augmented with simple translation and complex rotation without visibility
- 1.4 : Matrix augmented with complex rotations ( no translation )
- 1.5 : Matrix augmented with complex rotations with video augmented
    -> Achieved 68% accuracy

on 10 words
- 1.6 / 2.0 : Matrix augmented with complex rotation for 10 epochs 
- 1.7 / 2.1 : Matrix augmented with complex rotation for 25 epochs 
- 1.8 / 2.2 : Matrix augmented with complex rotation on 36 frames
    -> Model's performance didn't improve
- 1.9 / 2.3 : Own videos with matrix aug 48 frames
- 2.0 / 2.4 : Own videos plus WL-ASL dataset with matrix augmentation 48 frames
    -> 88% accuracy on Nick live test videos
- 2.1 : Only WLASL videos
    -> Model couldn't converge, not enough data
- 2.2 : 
- 2.3 : 
TODO: Describe 2.2, 2.3

on 20 words
- 2.4 / 3.0 : Only Own plus WLASL for 250 epochs
    -> Model underfits, needs more data
- 2.5 / 3.1 : 
    -> 81% v2.5_14 (14th epoch)
- 2.6 / 3.2: Only Own plus WLASL for 350 epochs 
    -> Doesn't converge
- 2.7 / 3.3: Own plus WLASL with 5 diff temporal fits
    -> Converges near 123, 130
- 2.8 / 3.4 : WLASL only with 5 diff temporal fits

- 2.9 / 3.5 : WLASL only with 5 diff temporal fits + single matrix aug with 3 diff temporal fits
'''