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

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset_loader import SignLanguageGestureDataset
from asl_model import AslNeuralNetwork

from sklearn.metrics import confusion_matrix
import seaborn as sns

# -- Define hyper parameters -- #

# TODO: Determine batch size and num epochs
# Optimal batch size: 64 (Must be divisible by 8) -> 512
# Optimal learning rate: Bt 0.0001 and 0.01 (Default: 0.001)
NUM_RNN_LAYERS = 4
NUM_EPOCHS = 100
BATCH_SIZE = 1024 #128 #2048
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.006 #0.006 #0.01

MODEL_PATH = "../../models"
# LOAD_MODEL_VERSION = "v4.7_31.5_22_selu_0.25"
NEW_MODEL_VERSION = "v5.01"
if not os.path.exists(f'{MODEL_PATH}/{NEW_MODEL_VERSION}'):
    os.makedirs(f'{MODEL_PATH}/{NEW_MODEL_VERSION}')

# Create model
model = AslNeuralNetwork(num_lstm_layers=NUM_RNN_LAYERS)

# Define loss, optimizer and lr scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# dataset_aug, 0.01, 1, 0.9
# dataset_rot_only, 0.06, 1, 0.9
# dataset_no_aug, 0.01, 5, 0.9
# dataset_no_aug_no_vis, 0.006, 5, 0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

# -- Load Model -- #
# model_state_dict = torch.load(f'{MODEL_PATH}/asl_model_{LOAD_MODEL_VERSION}.pth')
# model.load_state_dict(model_state_dict)
# optimizer_state_dict = torch.load(f'{MODEL_PATH}/asl_optimizer_{LOAD_MODEL_VERSION}.pth')
# optimizer.load_state_dict(optimizer_state_dict)

# -- Constants -- #
DATASET_PATH = '../../inputs/dataset_rot_only_no_vis'
TEST_PATH = '../../processed_tests/joint_no_vis'
DATASET_FILES = os.listdir(DATASET_PATH)
TEST_FILES = os.listdir(ALI_TEST_PATH)
DATASET_SIZE = len(DATASET_FILES)


# Define signed words/action classes 
word_dict = {'bad': 0, 'bye': 1, 'easy': 2, 'good': 3, 'happy': 4, 'hello': 5, 'like': 6, 'me': 7, 'meet': 8, 'more': 9, 'no': 10, 'please': 11, 'sad': 12, 'she': 13, 'sorry': 14, 'thank you': 15, 'want': 16, 'why': 17, 'yes': 18, 'you': 19}

# -- Load dataset -- #
TRAIN_SPLIT = int(DATASET_SIZE * 0.85)
VALID_SPLIT = DATASET_SIZE - TRAIN_SPLIT

dataset = SignLanguageGestureDataset(DATASET_PATH, DATASET_FILES, word_dict)
train_split, valid_split = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, VALID_SPLIT])

train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True)

test_set = SignLanguageGestureDataset(TEST_PATH, TEST_FILES, word_dict)
test_loader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=True)

train_size = len(train_loader)
valid_size = len(valid_loader)

no_train_loss_items = int(train_size/10) #30
no_valid_loss_items = int(valid_size/5) #10

# Create summary writer for Tensorboard
writer = SummaryWriter()

# -- Train Model -- #
patience=3
patience_counter = 0
best_epoch_loss = 3
best_epoch_accuracy = 0
for epoch in range(NUM_EPOCHS):
    '''
    # Forward pass
    output = model(keypoints)
    loss = criterion(output, labels)
    
    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute training loss
    print(epoch, loss.item())
    scheduler.step()
    '''

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
        
    
    # -- Quick Test -- #
    true_count = 0
    test_count = 0
    for i, (keypoints, labels) in enumerate(valid_loader):  
        # Use GPU for model validation computations  
        keypoints = keypoints.to(model.device)
        labels = labels.to(model.device)

        # Get prediction
        output = model(keypoints)

        # Obtain max prob value and class index
        _, predicted = torch.max(output.data, 1)
        
        true_count += (predicted == labels).sum().item()
        test_count += labels.size(0)

    model_accuracy = (true_count / test_count) * 100
    print(f'Model Accuracy: {model_accuracy:.2f}%')
    
    if best_epoch_accuracy < model_accuracy:
        torch.save(model.state_dict(), f'{MODEL_PATH}/{NEW_MODEL_VERSION}/asl_model_{NEW_MODEL_VERSION}_acc={model_accuracy:.2f}_epo={epoch}.pth')
        best_epoch_accuracy = model_accuracy
        print(f'->Model Saved w/ Best Accuracy<-')

    # Save model with early stopping to prevent overfitting
    # Condition: Better validation loss in under patience epochs
    train_epoch_loss = train_total_loss/train_size
    valid_epoch_loss = valid_total_loss/valid_size
    if valid_epoch_loss < best_epoch_loss:
        best_epoch_loss = valid_epoch_loss
        
        patience_counter = 0
        
        torch.save(model.state_dict(), f'{MODEL_PATH}/{NEW_MODEL_VERSION}/asl_model_{NEW_MODEL_VERSION}.pth')
        print(f'Model Saved - Train={train_epoch_loss}  Valid={valid_epoch_loss}')
    else:
        patience_counter += 1

        if patience_counter >= patience:
            break

    # Track training and validation loss in Tensorboard
    writer.add_scalars(f'Loss Plot - nn_{NEW_MODEL_VERSION}', 
        { 
            'train_loss': train_total_loss/train_size,
            'valid_loss': valid_total_loss/valid_size,
        }, 
        epoch 
    )
        
    scheduler.step()


# View tensorboard using following command:
#tensorboard --logdir runs
writer.close()