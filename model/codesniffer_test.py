import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from model.modules.sniffer import CodeSnifferNetwork
import torch
import pandas as pd

random.seed(90)
BATCH_SIZE = 16
WORKERS = 4
NUM_LABELS=8
PATH = "models/codeSniffer.pth"
LABELS_FILE = "../data/filtered_data.csv"

dataset = CodeSnifferDataset('../data/filtered_data.csv', '../data/code_files')

classes = pd.read_csv(LABELS_FILE).iloc[0]

classes = classes.index.to_list()[1:]

print("Instantiated dataset")

ind = random.randrange(len(dataset))

input_ids, attention_mask, labels = dataset[ind]

print("Got one tensor")


model = CodeSnifferNetwork(num_labels=NUM_LABELS)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    print("One tensor: ")
    y_pred = model.forward(input_ids, attention_mask)
    y_pred = y_pred.squeeze(0).tolist()
    y_pred = [round(num, 2) for num in y_pred]
    y_pred = dict(zip(classes, y_pred))
    print(f"y_pred = {y_pred}")
    y_true = labels.tolist()
    y_true = dict(zip(classes, y_true))
    print(f"y_true = {y_true}")