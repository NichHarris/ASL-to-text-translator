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

import os

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.samplers import TPESampler
# pip3.10 install optuna optuna-dashboard
# pip3.10 install mysql-connector-python

# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
NUM_EPOCHS = 60
BATCH_SIZE = 64
VALID_BATCH_SIZE = 8

MODEL_PATH = "../../smbo"

# -- Constants -- #
DATASET_PATH = '../../inputs/dataset_no_aug_no_vis'
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)

VALID_SET_PATH= '../../processed_tests/joint_no_vis'
VALID_SET_FILES = os.listdir(VALID_SET_PATH)
VALID_SET_SIZE = len(VALID_SET_FILES)
print(DATASET_SIZE, VALID_SET_SIZE)

# Define signed words/action classes 
word_dict = {'bad': 0, 'bye': 1, 'easy': 2, 'good': 3, 'happy': 4, 'hello': 5, 'like': 6, 'me': 7, 'meet': 8, 'more': 9, 'no': 10, 'please': 11, 'sad': 12, 'she': 13, 'sorry': 14, 'thank you': 15, 'want': 16, 'why': 17, 'yes': 18, 'you': 19}

# -- Load dataset -- #
train_set = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

valid_set = SignLanguageGestureDataset(VALID_SET_PATH, VALID_SET_FILES, word_dict)
valid_loader = DataLoader(dataset=valid_set, batch_size=VALID_BATCH_SIZE, shuffle=True)

train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = int(train_size/5)
no_valid_loss_items = int(valid_size/5)

# -- Train Model -- #
def train_model(trial, params):
    # Create model
    model = AslNeuralNetwork(lstm_hidden_size=params['lstm_hidden_size'], num_lstm_layers=params['num_lstm_layers'], dropout_rate=params['dropout'])
    print(f"{params['num_lstm_layers']} x {params['lstm_hidden_size']} biLSTM w/ lr {params['learning_rate']} + dr {params['dropout']}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'])

    patience = 5
    patience_counter = 0
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

            patience_counter = 0
            torch.save(model.state_dict(), f'{MODEL_PATH}/{trial.number}/asl_model_{trial.number}_vl={valid_epoch_loss}.pth')
            print(f'Model Saved - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
        else:
            patience_counter += 1

            if patience_counter > patience:
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
        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 128, 256), #512
        'learning_rate':  trial.suggest_float('learning_rate', 1e-4, 1e-1),
        'batch_size': trial.suggest_int('batch_size', 64, 1024),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5) 
    }
    # TODO: Consider num fc nodes later
    # 'fc_hidden_size': trial.suggest_int('fc_hidden_size', 64, 128), # 256

    if not os.path.exists(f'{MODEL_PATH}/{trial.number}'):
        os.makedirs(f'{MODEL_PATH}/{trial.number}')
    
    # Train with params
    best_loss = train_model(trial, params)

    return best_loss
    
# Create study minimizing validation loss or maximize accuracy
# TODO: Look into how to setup mysql
study = optuna.create_study(
    direction='minimize',
    study_name='top_20_words',
    sampler=TPESampler()
    # storage="mysql://root@127.0.0.1/optunadb", 
    # load_if_exists= True,
    # sampler= ..., pruner=...
)
study.optimize(objective, n_trials=100)

trialx = study.best_trial
print(trialx.values)
print(trialx.params)

# TODO: Integrate Raytune with Optuna 


# brew install mysql
# mysql -u root
# CREATE DATABASE optunadb;