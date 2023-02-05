'''
Optuna - Hyperparameter optimization for Pytorch
https://www.youtube.com/watch?v=P6NwZVl8ttc

- Sampling Strategy
    -> Finds best paths with Bayesian filtering
    - Sampler - 
        TPE (Tree-structured Parzen Estimator)
        GP (Gaussian Process), 
        CMA-ES (Covariance matrix adaptation evolution strategy) 
        
        Under 1000 trials
            w/ correlated parameters -> GP
            w/ not correlated parameters -> TPE
        
        Over 1000 trials -> CMA-ES
- Pruning Strategy 
    -> End unpromising trials early
    - Integrated -
        Pytorch Lightning, Ignite
        FastAI

"Asynchronous parallelization of trials with near-linear scaling"

Visualization
Provides importance of hyperparameters
- learning rates
- num units in first layer
- dropout first layer
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms

import os

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import optuna
# pip3.10 install optuna optuna-dashboard

# pip3.10 install mysql-connector-python

# Define hyper parameters
INPUT_SIZE = 226 # 226 datapoints from 67 landmarks - 21 in x,y,z per hand and 25 in x,y,z, visibility for pose
SEQUENCE_LEN = 48 # 48 frames per video
OUTPUT_SIZE = 20 # Starting with 5 classes = len(word_dict) # TODO: Get from word_dict size


# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
NUM_EPOCHS = 30
BATCH_SIZE = 128 #1024

MODEL_PATH = "../../smbo"

# -- Constants -- #
DATASET_PATH = '../../inputs/dataset_no_aug'
# '../../inputs/dataset_only'
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)

VALID_SET_PATH= '../../processed_tests/ali'
VALID_SET_FILES = os.listdir(VALID_SET_PATH)
VALID_SET_SIZE = len(VALID_SET_FILES)
print(DATASET_SIZE, VALID_SET_SIZE)

# Define signed words/action classes 
word_dict = {'bad': 0, 'bye': 1, 'easy': 2, 'good': 3, 'happy': 4, 'hello': 5, 'like': 6, 'me': 7, 'meet': 8, 'more': 9, 'no': 10, 'please': 11, 'sad': 12, 'she': 13, 'sorry': 14, 'thank you': 15, 'want': 16, 'why': 17, 'yes': 18, 'you': 19}

# -- Load dataset -- #
train_set = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

valid_set = SignLanguageGestureDataset(VALID_SET_PATH, VALID_SET_FILES, word_dict)
valid_loader = DataLoader(dataset=valid_set, batch_size=32, shuffle=True)

train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = int(train_size/3)
no_valid_loss_items = int(valid_size/3)

# -- Train Model -- #
def train_model(trial, params):
    # Create model
    model = AslNeuralNetwork(INPUT_SIZE, params['lstm_hidden_size'], params['fc_hidden_size'], OUTPUT_SIZE, params['num_lstm_layers'])
    print(params['lstm_hidden_size'], params['fc_hidden_size'], params['num_lstm_layers'], params['learning_rate'])

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'])

    patience=3
    best_epoch_loss = 100
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
        
        # Early Stopping + Save model with improved validation loss
        train_epoch_loss = train_total_loss/train_size
        valid_epoch_loss = valid_total_loss/valid_size
        if valid_epoch_loss < best_epoch_loss:
            best_epoch_loss = valid_epoch_loss

            patience = 3
            has_patience_started = True
            if valid_epoch_loss < 0.2:
                torch.save(model.state_dict(), f'{MODEL_PATH}/asl_model_{trial.number}_vl={valid_epoch_loss}.pth')
            print(f'Model Saved - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
        else:
            patience -= 1

            if patience == 0:
                print(f'Early Stopping - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
                return best_epoch_loss
        
        # -- Pruning Strategy -- 
        # Evaluate trial based on performance relative to time/depth
        # to terminate or continue trial
        trial.report(best_epoch_loss, epoch)
        if trial.should_prune(): 
            raise optuna.exceptions.TrialPruned()

    return best_epoch_loss


def objective(trial):
    params = {
        'num_lstm_layers': trial.suggest_int('num_lstm_layers', 3, 6), # 8
        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 64, 256), #512
        'fc_hidden_size': trial.suggest_int('fc_hidden_size', 32, 128), # 256
        'learning_rate':  trial.suggest_float('learning_rate', 1e-4, 1e-1),
    }
    # 'dropout': trial.suggest_uniform('dropout', 0.1, 0.8)  
    
    # Train with params
    best_loss = train_model(trial, params)

    return best_loss
    
# Create study minimizing loss
# TODO: Look into how to setup mysql
study = optuna.create_study(
    direction='minimize',
    study_name='top_20_words',
    # storage="mysql://root@127.0.0.1/optunadb", 
    # load_if_exists= True,
    # sampler= ..., pruner=...
)
study.optimize(objective, n_trials=20)

trialx = study.best_trial
print(trialx.values)
print(trialx.params)


# brew install mysql
# mysql -u root
# CREATE DATABASE optunadb;