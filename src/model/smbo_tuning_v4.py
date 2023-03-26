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
from torch.cuda.amp import autocast, GradScaler
from ray.tune.search.bohb import TuneBOHB

import os

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from torch.utils.data import DataLoader
from ray.tune.search.bohb import TuneBOHB
from torch.utils.tensorboard import SummaryWriter
# pip3.10 install optuna optuna-dashboard "ray[default]" "ray[air]" "ray[tune]" bayesian-optimization

# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
NUM_EPOCHS = 100
BATCH_SIZE = 512
TEST_BATCH_SIZE = 8

# -- Constants -- #
DATASET_NAME = 'dataset_next_42_rot_only_no_vis'
DATASET_PATH = os.path.abspath(f'../../inputs/dataset_next_42_rot_only_no_vis')
DATASET_FILES = os.listdir(DATASET_PATH)
DATASET_SIZE = len(DATASET_FILES)
print(DATASET_SIZE,'------------------------------------------------------')

TEST_SET_PATH= os.path.abspath('../../tests/joint_no_vis')
TEST_SET_FILES = os.listdir(TEST_SET_PATH)
TEST_SET_SIZE = len(TEST_SET_FILES)

# Define signed words/action classes 
word_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'afternoon': 36, 'answer': 37, 'bad': 38, 'big': 39, 'buy': 40, 'bye': 41, 'can': 42, 'day': 43, 'easy': 44, 'evening': 45, 'excuse': 46, 'forget': 47, 'give': 48, 'good': 49, 'happy': 50, 'hear': 51, 'hello': 52, 'here': 53, 'how': 54, 'know': 55, 'left': 56, 'like': 57, 'love': 58, 
'me': 59, 'meet': 60, 'month': 61, 'more': 62, 'morning': 63, 'name': 64, 'night': 65, 'no': 66, 'out': 67, 'please': 68, 'question': 69, 'read': 70, 
'remember': 71, 'right': 72, 'sad': 73, 'see': 74, 'sell': 75, 'she': 76, 'small': 77, 'sorry': 78, 'take': 79, 'thank you': 80, 'think': 81, 'time': 
82, 'today': 83, 'tomorrow': 84, 'understand': 85, 'want': 86, 'week': 87, 'what': 88, 'when': 89, 'where': 90, 'which': 91, 'who': 92, 'why': 93, 'with': 94, 'write': 95, 'wrong': 96, 'yes': 97, 'yesterday': 98, 'you': 99}

# -- Load dataset -- #
TRAIN_SPLIT = int(DATASET_SIZE * 0.9)
VALID_SPLIT = DATASET_SIZE - TRAIN_SPLIT

dataset = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)

# Split into training, validation and testing sets
train_split, valid_split = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALID_SPLIT])

train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True)
#train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True ,pin_memory=True)
#valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

test_set = SignLanguageGestureDataset(TEST_SET_PATH, TEST_SET_FILES, word_dict)
test_loader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=True,  pin_memory=True)
train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = int(train_size/5)
no_valid_loss_items = int(valid_size/3)

def objective(params):
    # Create model
    model = AslNeuralNetwork(lstm_hidden_size=int(params['lstm_hidden_size']), num_lstm_layers=int(params['num_lstm_layers']), dropout_rate=params['dropout'])
    print(f"{params['num_lstm_layers']} x {params['lstm_hidden_size']} biLSTM w/ lr {params['learning_rate']} + dr {params['dropout']}")
    model.to(model.device, non_blocking=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=params['learning_rate'])
     
    patience = 15
    patience_counter = 0
    best_epoch_loss = 100
    best_epoch_accuracy = 0
    scaler =GradScaler()
    cache_limit = 10
    for epoch in range(NUM_EPOCHS):
        print(f'--- Starting Epoch #{epoch} ---')

        # -- Actual Training -- #
        mini_batch_loss = 0
        train_total_loss = 0
        for i, (keypoints, labels) in enumerate(train_loader):
            # Use GPU for model training computations  
            keypoints = keypoints.to(model.device)
            labels = labels.to(model.device)


            with torch.autocast(device_type= 'cuda', dtype=torch.float16):
                # Forward pass
                output = model(keypoints)
                loss = criterion(output, labels)

            # Back propagation
            #loss.backward()
            #optimizer.step()

            # Back propagation
            optimizer.zero_grad()
            # for loop faster the zero_grad
            #for param in model.parameters():
            #    param.grad = None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Compute training loss
            train_total_loss += loss.item()
            # mini_batch_loss += loss.item()
            # if (i + 1) % no_train_loss_items == 0:
            #     print(f"Training Loss - {i + 1}/{train_size}: {mini_batch_loss/no_train_loss_items}")
            #     mini_batch_loss = 0

            #empty gpu cache       
            if (i + 1) % cache_limit == 0:
                torch.cuda.empty_cache()    

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
        
        #session.report({"model_accuracy": model_accuracy})
        # Track training and validation loss in Tensorboard
        session.report({'validation_loss': valid_epoch_loss})
    

#utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
# Bayesian Optimization Search with Gaussian Process and UCB (Upper confidence bound)
#algo = BayesOptSearch()
algo = TuneBOHB()
algo = ConcurrencyLimiter(algo, max_concurrent=8)

TUNER_METRIC = 'validation_loss' #"model_accuracy" #
TUNER_MODE = 'min'#"max" #
log_file_name = f'logs/{DATASET_NAME}_{TUNER_MODE}_{TUNER_METRIC}.log'

# Specify resources
# each trial gets the below mentioned gpu and cpu
trainable_with_resources = tune.with_resources(
    objective,
    {
        'gpu': 0.5,
        'cpu': 2
    }
)

# Integrate Raytune with Optuna 
tuner = tune.Tuner(
    trainable_with_resources,
    run_config=air.RunConfig(log_to_file=True),
    tune_config=tune.TuneConfig(
        metric=TUNER_METRIC,
        mode=TUNER_MODE,
        search_alg=algo,
        num_samples=150
    ),
    # run_config=air.RunConfig(
    #     name="smbo_raytune_opt_min_v1",
    # ),
    param_space={
        'num_lstm_layers':tune.choice([3,4,5,6]), #tune.choice([3, 4, 5, 6]),
        'lstm_hidden_size':tune.choice([126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256]), #tune.choice([128, 256]),
        'learning_rate':  tune.uniform(1e-3, 1e-1),
        'dropout': tune.choice([0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]) 
    }
)

res = tuner.fit()
best_trial = res.get_best_result()

print("Best trial: ", best_trial)
print("Best hyperparameters found were: ", best_trial.config)


with open(log_file_name, 'w') as log_file: 
    log_file.write(f"Best trial: {best_trial}\n" )
    log_file.write(f"Best hyperparameters found were: {best_trial.config}\n" )

