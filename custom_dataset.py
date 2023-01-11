import os
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset: https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

class SignLanguageGestureDataset(Dataset):
    def __init__(self, data_path, files_list, sign_dict):
        self.root = data_path
        self.files = files_list
        self.signs = sign_dict

    def __len__(self):
        # Length of dataset
        return len(self.files)
    
    def __getitem__(self, index):
        # Data instance and label tuple
        filename = self.files[index]
        
        # Load data from file
        data = torch.load(f'{self.root}/{filename}')

        # Convert label from filename to target value
        label = filename.split('_')[0]
        target = self.signs[label]

        return (data, target)


'''
# -- Old method for loading data --- #

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