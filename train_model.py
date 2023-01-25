
# RNN Tutorial : https://www.youtube.com/watch?v=0_PgWWmauHk


# Training with image sequence (ie video) 
# Standard video frame rate is 24fps

'''
Video: Temporal information (Sequence of frames, Sequential data)

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


'''
https://www.youtube.com/watch?v=bA7-DEtYCNM
PyTorch Deployment on Flask
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

from custom_dataset import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork, device

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define hyper parameters
INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 3 # 3 LSTM Layers

LSTM_HIDDEN_SIZE = 128 # 128 nodes in LSTM hidden layers
FC_HIDDEN_SIZE = 64 # 64 nodes in Fc hidden layers
OUTPUT_SIZE = 5 # Starting with 5 classes = len(word_dict)

# TODO: Determine batch size and num epochs
LEARNING_RATE = 0.001
BATCH_SIZE = 100
NUM_EPOCHS = 33 # TODO: Replace back with 100

MODEL_PATH = "./model"
LOAD_MODEL_VERSION = "v1.0"
NEW_MODEL_VERSION = "v1.5"

# Create model
model = AslNeuralNetwork(INPUT_SIZE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# # -- Load Model -- #
# model_state_dict = torch.load(f'{MODEL_PATH}/asl_model_{LOAD_MODEL_VERSION}.pth')
# optimizer_state_dict = torch.load(f'{MODEL_PATH}/asl_optimizer_{LOAD_MODEL_VERSION}.pth')
# model.load_state_dict(model_state_dict)
# optimizer.load_state_dict(optimizer_state_dict)

# -- Constants -- #
VIDEOS_PATH = './data'
DATASET_PATH = './dataset5'
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)

TRAIN_SPLIT = int(DATASET_SIZE * 0.9)
TEST_SPLIT = DATASET_SIZE - TRAIN_SPLIT #(DATASET_SIZE * 0.1)

ACTUAL_TRAIN_SPLIT = int(TRAIN_SPLIT * 0.8)
VALID_SPLIT = TRAIN_SPLIT - ACTUAL_TRAIN_SPLIT #(TRAIN_SPLIT * 0.2)

# Define signed words/action classes 
word_dict = {}
for i, sign in enumerate(os.listdir(VIDEOS_PATH)):
    word_dict[sign] = i

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

no_train_loss_items = 5
no_valid_loss_items = 2

# -- Train Model -- #
for epoch in range(NUM_EPOCHS):
    print(f'--- Starting Epoch #{epoch} ---')

    # -- Actual Training -- #
    train_loss = 0
    for i, (keypoints, labels) in enumerate(train_loader):
        # Use GPU for model training computations  
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, labels)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training loss
        train_loss += loss.item()
        if (i + 1) % no_train_loss_items == 0:
            print(f"Training Loss on {i + 1}/{train_size}: {train_loss/no_train_loss_items}")
            train_loss = 0


    # -- Validation -- #
    # TODO: Determine what else to do in validation section
    valid_loss = 0
    for i, (keypoints, labels) in enumerate(valid_loader):  
        # Use GPU for model validation computations  
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, labels)
        
        # Compute validation loss
        valid_loss += loss.item()
        if (i + 1) % no_valid_loss_items == 0:
            print(f'Validation Loss {i + 1}/{valid_size}: {valid_loss/no_valid_loss_items}')
            valid_loss = 0
        
        # TODO: valid loss < training loss -> save model
    
    # TODO: Add graph to track training and validation loss

# -- Test Model -- #
# Gradient computation not needed in testing phase
with torch.no_grad():
    # TODO; Convert to f1-measure instead of accuracy
    true_count = 0
    test_count = 0
    for keypoints, labels in test_loader:
        # Use GPU for model testing computations  
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        
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


# -- Trying to plot in Tensorboard -- #
# writer = SummaryWriter()
# for i in range(10):
#     writer.add_scalar('Train Loss', i, i)
# writer.close()