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
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

import os

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from torch.utils.data import DataLoader

# pip3.10 install optuna optuna-dashboard "ray[default]" "ray[air]" "ray[tune]" bayesian-optimization

# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
NUM_EPOCHS = 120
BATCH_SIZE = 64
TEST_BATCH_SIZE = 8

# -- Constants -- #
DATASET_PATH = os.path.abspath('../../inputs/dataset_no_aug')
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)

TEST_SET_PATH= os.path.abspath('../../processed_tests/joint')
TEST_SET_FILES = os.listdir(TEST_SET_PATH)
TEST_SET_SIZE = len(TEST_SET_FILES)

# Define signed words/action classes 
word_dict = {'bad': 0, 'bye': 1, 'easy': 2, 'good': 3, 'happy': 4, 'hello': 5, 'like': 6, 'me': 7, 'meet': 8, 'more': 9, 'no': 10, 'please': 11, 'sad': 12, 'she': 13, 'sorry': 14, 'thank you': 15, 'want': 16, 'why': 17, 'yes': 18, 'you': 19}

# -- Load dataset -- #
TRAIN_SPLIT = int(DATASET_SIZE * 0.9)
VALID_SPLIT = DATASET_SIZE - TRAIN_SPLIT

dataset = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)

# Split into training, validation and testing sets
train_split, valid_split = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALID_SPLIT])

train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True)

test_set = SignLanguageGestureDataset(TEST_SET_PATH, TEST_SET_FILES, word_dict)
test_loader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=True)

train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = int(train_size/5)
no_valid_loss_items = int(valid_size/3)

def objective(params):
    # Create model
    model = AslNeuralNetwork(lstm_hidden_size=params['lstm_hidden_size'], num_lstm_layers=params['num_lstm_layers'], dropout_rate=params['dropout'])
    print(f"{params['num_lstm_layers']} x {params['lstm_hidden_size']} biLSTM w/ lr {params['learning_rate']} + dr {params['dropout']}")
    model.to(model.device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'])

    patience = 5
    patience_counter = 0
    best_epoch_loss = 100
    best_epoch_accuracy = 0
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
            # mini_batch_loss += loss.item()
            # if (i + 1) % no_train_loss_items == 0:
            #     print(f"Training Loss - {i + 1}/{train_size}: {mini_batch_loss/no_train_loss_items}")
            #     mini_batch_loss = 0

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
            # mini_batch_loss += loss.item()
            # if (i + 1) % no_valid_loss_items == 0:
            #     print(f'Validation Loss - {i + 1}/{valid_size}: {mini_batch_loss/no_valid_loss_items}')
            #     mini_batch_loss = 0
        
        # -- Quick Test -- #
        true_count = 0
        test_count = 0
        for i, (keypoints, labels) in enumerate(test_loader):  
            # Use GPU for model validation computations  
            keypoints = keypoints.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass
            output = model(keypoints)

            # Obtain max prob value and class index
            _, predicted = torch.max(output.data, 1)
            
            true_count += (predicted == labels).sum().item()
            test_count += labels.size(0)
            
        model_accuracy = (true_count / test_count) * 100
        print(f'Model Accuracy: {model_accuracy:.2f}%')

        if best_epoch_accuracy < model_accuracy:
            # torch.save(model.state_dict(), f'{MODEL_PATH}/{trial.number}/asl_model_{trial.number}_acc={model_accuracy:.2f}_epo={epoch}.pth')
            best_epoch_accuracy = model_accuracy
            print(f'->Model Saved w/ Best Accuracy<-')

        # Early Stopping + Save model with improved validation loss
        train_epoch_loss = train_total_loss/train_size
        valid_epoch_loss = valid_total_loss/valid_size
        if valid_epoch_loss < best_epoch_loss:
            best_epoch_loss = valid_epoch_loss

            patience_counter = 0
            # torch.save(model.state_dict(), f'{MODEL_PATH}/{trial.number}/asl_model_{trial.number}_vl.pth')
            print(f'Model Saved - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
        else:
            patience_counter += 1

            if patience_counter > patience:
                print(f'Early Stopping - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
                break
        
        # session.report({"model_accuracy": model_accuracy})
        session.report({'validation_loss': valid_epoch_loss})


#utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
# Bayesian Optimization Search with Gaussian Process and UCB (Upper confidence bound)
algo = BayesOptSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)

# Integrate Raytune with Optuna 
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric='validation_loss',#"model_accuracy", #
        mode='min',#"max", #
        search_alg=algo,
        num_samples=4
    ),
    # run_config=air.RunConfig(
    #     name="smbo_raytune_opt_min_v1",
    # ),
    param_space={
        'num_lstm_layers': 4, #tune.choice([3, 4, 5, 6]),
        'lstm_hidden_size': 128, #tune.choice([128, 256]),
        'learning_rate':  tune.uniform(1e-4, 1e-1),
        'dropout': tune.uniform(0.2, 0.8) 
    }
)

res = tuner.fit()
best_trial = res.get_best_result()

print("Best trial: ", best_trial)
print("Best hyperparameters found were: ", best_trial.config)

'''
trainable_with_resources = tune.with_resources(
    trainable,
    {
        'gpu': 1,
        'cpu': 4
    }
)
'''