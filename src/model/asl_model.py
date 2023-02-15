import torch
import torch.nn as nn

# PyTorch LSTM Model : https://www.youtube.com/watch?v=0_PgWWmauHk
# PyTorch Deployment Flask : https://www.youtube.com/watch?v=bA7-DEtYCNM

# -- In and Out -- #
# 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
# 20 signs/classes currently recognized by model (out of 200)
# TODO: Could be reduced to 201 by ignoring visibility and increase to 40
INPUT_SIZE = 201 #201 trying no visibility instead of 226 
OUTPUT_SIZE = 20
# SEQUENCE_LEN = 48

# -- Neurons -- #
# Default 4 biLSTM layers
# 128 nodes in biLSTM hidden layers
# 64 nodes in Fc hidden layers
# TODO: Hyperparam Optimization: 3-6 layers and 64-256 for lstm + 2-4 layers and 32-128 for fc
NUM_LSTM_LAYERS = 4
LSTM_HIDDEN_SIZE = 128
FC_HIDDEN_SIZE = 64 
DROP_PROB = 0.5

class AslNeuralNetwork(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, lstm_hidden_size=LSTM_HIDDEN_SIZE, fc_hidden_size=FC_HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_lstm_layers=NUM_LSTM_LAYERS, dropout_rate=DROP_PROB):
        # Call Neural network module initialization
        super(AslNeuralNetwork, self).__init__()

        # Define constants
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size
        
        # Create device with gpu support
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define neural network architecture and activiation function
        # Long Short Term Memory (Lstm) and Fully Connected (Fc) Layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, self.num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, output_size)
        '''
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, output_size)
        '''

        # TODO: Add input normalization before rnn

        # Scaled Exponential Linear Units (SELU) Activation
        # -> Self-normalization (internal normalization) by converging to mean and unit variance of zero
        self.relu = nn.SELU() #nn.LeakyReLU() #
        self.dropout = nn.AlphaDropout(dropout_rate) #nn.Dropout(dropout_rate) #
        
    # Define forward pass, passing input x
    def forward(self, x):
        # Define initial tensors for hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_lstm_layers * 2, batch_size, self.lstm_hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_lstm_layers * 2, batch_size, self.lstm_hidden_size).to(self.device) 
        
        # Pass input with initial tensors to lstm layers
        out_lstm, _ = self.lstm(x, (h0, c0))
        out_relu = self.relu(out_lstm)
        
        # Many-One Architecture: Pass only last timestep to fc layers
        in_fc1 = out_relu[:, -1, :]
        in_fc1 = self.dropout(in_fc1)
        
        out_fc1 = self.fc1(in_fc1)
        in_fc2 = self.relu(out_fc1)
        in_fc2 = self.dropout(in_fc2)

        out = self.fc2(in_fc2)
        '''
        out_fc2 = self.fc2(in_fc2)
        in_fc3 = self.relu(out_fc2)
    
        # Note: Softmax already included in cross entropy loss function
        out = self.fc3(in_fc3)
        '''

        return out