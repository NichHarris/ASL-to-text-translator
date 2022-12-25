import torch
from torch.utils.data import Dataset
import pandas as pd

# Custom dataset: https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

class SignLanguageGestureDataset(Dataset):
    def __init__(self, file):
        self.annotations = pd.read_csv(file)