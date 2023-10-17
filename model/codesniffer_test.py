import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from torch.utils.data import DataLoader
from model.modules.sniffer import CodeSnifferNetwork
import torch

random.seed(90)
batch_size = 16
workers = 4

dataset = CodeSnifferDataset('../data/filtered_data.csv', '../data/code_files')

print("Instantiated dataset")

ind = random.randrange(len(dataset))

input_ids, attention_mask, labels = dataset[ind]

print("Got one tensor")

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
INPUT_IDS, ATTENTION_MASKS, LABELS = next(iter(data_loader))

print("Got batch")

model = CodeSnifferNetwork()
model.eval()
with torch.no_grad():
    print("One tensor: ")
    y_pred = model.forward(input_ids, attention_mask)
    print(y_pred)

    print("Batch: ")
    Y_PRED = model.forward(INPUT_IDS, ATTENTION_MASKS)
    print(Y_PRED)