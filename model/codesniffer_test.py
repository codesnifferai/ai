import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from torch.utils.data import DataLoader
from sniffer import CodeSnifferNetwork
import torch

random.seed(90)
batch_size = 16
workers = 4

dataset = CodeSnifferDataset('../data/filtered_data.csv', '../data/code_files')

ind = random.randrange(len(dataset))

x, y = dataset[ind]

X, Y = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

model = CodeSnifferNetwork()
model.eval()
with torch.nograd():
    print("One tensor: ")
    y_pred = model.forward(x)
    print(y_pred)

    print("Batch: ")
    Y_PRED = model.forward(X)