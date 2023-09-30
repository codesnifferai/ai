import os
import torch
import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import Dataset

class CodeSnifferDataset(Dataset):
    def __init__(self, annotations_file, code_files_dir, transform=None, target_transform=None):
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        self.code_labels = pd.read_csv(annotations_file)
        self.code_files_dir = code_files_dir
        self.transform = transform
        self.target_transform = target_transform
        self.tokenize_data()
        self.tokenized_data = torch.load(self.tokenized_data_path)


    def tokenize_data(self):
         # Get the directory of the current script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
        # Join the directory with the filename to get the full path
        self.tokenized_data_path = os.path.join(current_script_dir, "tokenized_data.pt")
        
        if os.path.isfile(self.tokenized_data_path):
            return
        
        tokenized_data = []
        for idx in range(len(self.code_labels)):
            code_file_path = os.path.join(self.code_files_dir, self.code_labels.iloc[idx, 0])
            with open(code_file_path, "r") as code_file:   # open text file in read mode
                code_data = code_file.read()               # read whole file to a string
                if self.transform:
                    code_data = self.transform(code_data)
                input = self.tokenizer(code_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tokenized_data.append(input)
            
        torch.save(tokenized_data, self.tokenized_data_path)
       
            

    def __len__(self):
        return len(self.code_labels)

    def __getitem__(self, idx):
        input_ids = self.tokenized_data[idx]["input_ids"]
        attention_mask = self.tokenized_data[idx]["attention_mask"]

        labels = self.code_labels.iloc[idx, 1:]
        labels = torch.tensor(labels, dtype=torch.int8)  # Convert to tensor
        return input_ids, attention_mask, labels