import torch
from torch.utils.data import Dataset

# PyTorch Dataset Loader: https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

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