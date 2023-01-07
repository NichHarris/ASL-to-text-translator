import torch
from torch.utils.data import Dataset

# Custom dataset: https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

class SignLanguageGestureDataset(Dataset):
    def __init__(self, data_path):
        self.root_path = data_path
        self.files = os.listdir(data_path)

    def __len__(self):
        # Length of dataset
        return len(self.files)
    
    def __getitem__(self, index):
        # Data instance and label tuple
        filename = self.files[index]
        
        data = torch.load(f'{self.root_path}/{filename}')
        label = filename.split('_')[0]

        return (data, label)


# -- Constants -- #
DATASET_PATH = './dataset'
DATASET_SIZE = 2180

TRAIN_SPLIT = DATASET_SIZE * 0.9
TEST_SPLIT = DATASET_SIZE * 0.1

ACTUAL_TRAIN_SPLIT = TRAIN_SPLIT * 0.8
VALID_SPLIT = TRAIN_SPLIT * 0.2

BATCH_SIZE = 10


# -- Load dataset -- #
dataset = SignLanguageGestureDataset(DATASET_PATH)

# Split into training, validation and testing sets
train_split, test_split = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT])
train_split, valid_split = torch.utils.data.random_split(train_split, [ACTUAL_TRAIN_SPLIT, VALID_SPLIT])

# Define train, valid, test data loaders
train_loader = DataLoader(dataset=train_split, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_split, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_split, batch_size=BATCH_SIZE, shuffle=True)

'''
# Train model
for id, (data, labels) in enumerate(train_loader): 

# Validate model
for id, (data, labels) in enumerate(valid_loader): 

# Test model
for id, (data, labels) in enumerate(test_loader): 
'''