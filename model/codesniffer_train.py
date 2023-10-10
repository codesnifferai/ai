import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sniffer import CodeSnifferNetwork
import torch
import torch.optim as optim
from train import train_model
import argparse

random.seed(90)


def main(args):
    train_percent = args.TRAIN_PERCENT
    batch_size = args.BATCH_SIZE
    lr = args.LEARNING_RATE
    num_labels = args.NUM_LABELS
    num_epochs = args.NUM_EPOCHS
    device = args.DEVICE
    annotations_file = args.ANNOTATIONS_FILE
    code_files_dir = args.CODE_FILES_DIR
    workers = args.WORKERS

    dataset = CodeSnifferDataset(annotations_file=annotations_file, code_files_dir=code_files_dir)

    train_size = int(train_percent * len(dataset))    # 80% for training and 20% for testing.
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    dataloaders = {"train": train_dataloader, "val": test_dataloader}

    codeModel = CodeSnifferNetwork(num_labels=num_labels)
    # codeModel = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(codeModel.parameters(), lr=lr)

    train_model(codeModel, dataloaders, criterion, optimizer, device, num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on some data.")
    
    # Add arguments
    # Add arguments
    parser.add_argument("--CODE_FILES", type=str, default="../data/code_files", help="Path to the code files")

    parser.add_argument("--ANNOTATIONS_FILE", type=str, default="../data/filtered_data.csv", help="Path to the annotations file")

    parser.add_argument("--DEVICE", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Device to run")
    
    parser.add_argument("--EPOCHS", type=int, default=300, help="Total number of training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("--WORKERS", type=int, default=4, help="Number of workers to dataloader")
    parser.add_argument("--TRAIN_PERCENT", type=float, default=0.8, help="Percentage of data to be used for training")
    parser.add_argument("--NUM_LABELS", type=int, default=8, help="Number of labels to be predicted")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args)