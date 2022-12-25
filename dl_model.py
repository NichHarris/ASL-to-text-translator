
# RNN Tutorial : https://www.youtube.com/watch?v=0_PgWWmauHk


# Training with image sequence (ie video) 
# Standard video frame rate is 24fps
#  

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

Transformers - Encoder-decoder architecture based on attention
- Sequences passed in parallel to reduce training time


Attention mechanism 
- Long reference window 
- Eg. Multi-Header Attention: Self Attention Mechanism 

Normalisation - ...

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

class AslNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden=2):
        # Call Neural network module initialization
        super(AslNeuralNetwork, self).__init__()

        # Define neural network architecture and activiation function
        self.rnn = nn.RNN(input_size, hidden_size, num_hidden, batch_first=True)

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.LSTM(hidden_size, output_size, num_hidden, batch_first=True, bidirectional=True) 
        self.output_layer = nn.Linear(output_size * 2, output_size)

        # Fc - Fully connected

        self.activation_hidden = nn.ReLU()


    # Define forward pass, passing input x
    def forward(self, x):
        first_out = self.input_layer(x)
        hidden_out = self.activation_hidden(first_out)
        return self.hidden_layer(hidden_out)
    
    # Define backward pass, minimizing error from output o
    def backward(self, o):
        pass

# Create device with gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyper parameters
INPUT_SIZE = 174 # 174 datapoints from 54 landmarks - 21 in x,y,z per hand and 12 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
NUM_RNN_LAYERS = 2 # 2 RNN Layers

HIDDEN_SIZE = 100
OUTPUT_SIZE = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 100

 
# Import dataset
X_train = ''

# Create model
model = AslNeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Train Model
for epoch in range(NUM_EPOCHS):
    for i, (keypoints, label) in enumerate(X_train):  
        # Forward pass
        output = model(keypoints)
        loss = criterion(output, label)
        
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verbose
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')



# Test Model
# torch.no_grad() # Gradient computation not needed in testing phase
