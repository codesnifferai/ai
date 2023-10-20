import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from model.modules.sniffer import CodeSnifferNetwork
import torch

random.seed(90)
BATCH_SIZE = 16
WORKERS = 4
NUM_LABELS=8
PATH = "models/codeSniffer.pth"

dataset = CodeSnifferDataset('../data/filtered_data.csv', '../data/code_files')

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
    print(f"y_pred = {y_pred}")
    labels = labels.tolist()
    print(f"y_true = {labels}")