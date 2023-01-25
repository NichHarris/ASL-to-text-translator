import torch
import torch.nn as nn

# PyTorch LSTM Model : https://www.youtube.com/watch?v=0_PgWWmauHk
# PyTorch Deployment Flask : https://www.youtube.com/watch?v=bA7-DEtYCNM

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
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, self.num_lstm_layers, batch_first=True) #bidirectional=True, dropout= ???
        self.fc1 = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.fc3 = nn.Linear(fc_hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        
        # Note: Softmax included in cross entropy loss function
        # self.softmax = nn.Softmax(dim=1)

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
    
        # Note: Softmax already included in cross entropy loss function
        out = self.fc3(in_fc3)
        # out = self.softmax(out_fc3)

        return out

# Create device with gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')