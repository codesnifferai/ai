import os
import pandas as pd
from torch.utils.data import Dataset

class CodeSnifferDataset(Dataset):
    def __init__(self, annotations_file, code_files_dir, transform=None, target_transform=None):
        self.code_labels = pd.read_csv(annotations_file)
        self.code_files_dir = code_files_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.code_labels)

    def __getitem__(self, idx):
        code_file_path = os.path.join(self.code_files_dir, self.code_labels.iloc[idx, 0])
        code_file = open(code_file_path, "r")      # open text file in read mode
        code_data = code_file.read()               # read whole file to a string
        code_file.close()                          # close file
        labels = self.code_labels.iloc[idx, 1:]
        if self.transform:
            code_data = self.transform(code_data)
        if self.target_transform:
            labels = self.target_transform(labels)
        return code_data, labels