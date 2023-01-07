
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

# Define hyper parameters
INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 3 # 3 LSTM Layers

LSTM_HIDDEN_SIZE = 128 # 128 nodes in LSTM hidden layers
FC_HIDDEN_SIZE = 64 # 64 nodes in Fc hidden layers
OUTPUT_SIZE = 5 # Starting with 5 classes

# TODO: Determine batch size and num epochs
LEARNING_RATE = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 100

class AslNeuralNetwork(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, fc_hidden_size, output_size):
        # Call Neural network module initialization
        super(AslNeuralNetwork, self).__init__()

        # Define constants
        self.num_lstm_layers = 3
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size

        # Define neural network architecture and activiation function
        # Long Short Term Memory (Lstm) and Fully Connected (Fc) Layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, self.num_rnn_layers, batch_first=True) #bidirectional=True, dropout= ???
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        # For bidirectional LSTMs
        # self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_hidden, batch_first=True, bidirectional=True) 
        # self.fc1 = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        
        # Look into dropout later
        # self.dropout = nn.Dropout()


    # Define forward pass, passing input x
    def forward(self, x):
        # Define initial tensors for hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(device) 
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size).to(device) 
        
        # Pass input with initial tensors to lstm layers
        out_lstm, _ = self.lstm(x, (h0, c0))
        out_relu = self.relu(out_lstm)
        
        # Many-One Architecture: Pass only last timestep to fc layers
        in_fc1 = out_relu[:, -1, :]
        out_fc1 = self.fc1(in_fc1)
        in_fc2 = self.relu(out_fc1)
        
        out_fc2 = self.fc2(in_fc2)
        in_fc3 = self.relu(out_fc1)
    
        out_fc3 = self.fc3(in_fc3)
        out = self.softmax(out_fc3)

        return out

# Create device with gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = AslNeuralNetwork(INPUT_SIZE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)



'''
# -- Train Model -- #
writer = SummaryWriter()
for epoch in range(NUM_EPOCHS):
    # -- Actual Training -- #
    train_loss = 0
    for i, (keypoints, label) in enumerate(train_loader):  
        #TODO: Check how to pass each frame for each instance
        # resized: [Batchsize, seqsize, inputsize]
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, label)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training loss
        train_loss += loss.item()
    
    # -- Validation -- #
    valid_loss = 0
    for i, (keypoints, label) in enumerate(valid_loader):  
        #TODO: Check how to pass each frame for each instance
        # resized: [Batchsize, seqsize, inputsize]
        keypoints = keypoints.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(keypoints)
        loss = criterion(output, label)
        
        # Compute validation loss
        valid_loss += loss.item()

    print(f"Validation Loss: {valid_loss}")
    print(f"Training Loss: {train_loss}")
    writer.add_scalar('Train Loss', train_loss)
    writer.add_scalar('Train Loss', train_loss)



# -- Test Model -- #
# Gradient computation not needed in testing phase
with torch.no_grad():
    # TODO; Variables for f1-measure
    for keypoints, labels in test_loader:
        # TODO: Format testing instances and labels
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        
        # Pass testing instances
        outputs = model(images)

        # Obtain max prob value and class index
        _, predicted = torch.max(outputs.data, 1)
        
        # TODO: Perform f1 measure
'''

# writer = SummaryWriter()
# for i in range(10):
#     writer.add_scalar('Train Loss', i, i)
# writer.close()

'''
# Load dataset
VIDEOS_PATH = "./data"
DATASET_PATH = "./dataset"

# Define signed words/action classes 
word_dict = {}
for i, sign in enumerate(os.listdir(VIDEOS_PATH)):
    word_dict[sign] = i


# Load dataset and labels
dataset_files = os.listdir(DATASET_PATH)
dataset_size = len(dataset_files)

labels = []
dataset = torch.zeros([dataset_size, 48, 226], dtype=torch.float)
for i, filename in enumerate(dataset_files):
    # Extract word from filename and obtain attributed index value
    word = filename.split("_")[0]
    index = word_dict[word]
    labels.append(index)

    # Load data instance and add to complete dataset 
    dataset[i] = torch.load(f'{DATASET_PATH}/{filename}')  
'''


# test_files = os.listdir("./dataset")[:15]

# temp_files = []
# for i, test_file in enumerate(test_files):
#     temp_files.append([])
#     for data in torch.load(f'./dataset/{test_file}'):
#         temp_data = []
#         for t in data:
#             temp_data.append(t.item())
        
#         temp_files[i].append(temp_data)


# # Dataset (Size, Sequences, Inputs)
# dataset = torch.tensor(temp_files, dtype=torch.float)
# print(dataset.size())

# # Split dataset into training, validation and testing

# # Split into batches to train and validate model
# batch_size = 10
# batched_data = torch.split(dataset, batch_size)
# print("SUCCESS")
# print(batched_data[0].size())
# print(batched_data[1].size())

# t1 = torch.zeros([48, 226], dtype=torch.float)
# for seq, data in enumerate(torch.load('test.pt')):
#     t1[seq] = data

# torch.save(t1, 'test.pt')
print("SUCCESS")
