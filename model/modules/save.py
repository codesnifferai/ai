import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_path(path):
    dir = os.path.dirname(path)
    if dir: 
        if not os.path.exists(dir):
            os.makedirs(dir)


def save_model(model, current_dir):
    # Save the Generator
    model_path = os.path.join(current_dir, "models/codeSniffer.pth")
    make_path(model_path)
    torch.save(model.state_dict(), model_path)

def save_statistics(current_dir, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist, best_true, best_preds):
    # Save the validation accuracy history
    val_acc_hist_np = np.array([h for h in val_acc_hist])
    val_acc_hist_path = os.path.join(current_dir, "statistics/val_acc_history.csv")
    make_path(val_acc_hist_path)
    np.savetxt(val_acc_hist_path, val_acc_hist_np, delimiter=",")

    #Save the validation loss history
    val_loss_hist_np = np.array([h for h in val_loss_hist])
    val_loss_hist_path = os.path.join(current_dir, "statistics/val_loss_history.csv")
    make_path(val_loss_hist_path)
    np.savetxt(val_loss_hist_path, val_loss_hist_np, delimiter=",")


    # Save the train accuracy history
    train_acc_hist_np = np.array([h for h in train_acc_hist])
    train_acc_hist_path = os.path.join(current_dir, "statistics/train_acc_history.csv")
    make_path(train_acc_hist_path)
    np.savetxt(train_acc_hist_path, train_acc_hist_np, delimiter=",")

    #Save the train loss history
    train_loss_hist_np = np.array([h for h in train_loss_hist])
    train_loss_hist_path = os.path.join(current_dir, "statistics/train_loss_history.csv")
    make_path(train_loss_hist_path)
    np.savetxt(train_loss_hist_path, train_loss_hist_np, delimiter=",")

    # Convert lists to DataFrames
    confusion_df = pd.DataFrame({'True': best_true, 'Predicted': best_preds})

    # Save to csv
    confusion_path = os.path.join(current_dir, "statistics/confusion.csv")
    make_path(confusion_path)
    confusion_df.to_csv(confusion_path, index=False)

def train_val_plot(current_dir, arrTrain, arrVal, title, stat):
    
    plot_path = os.path.join(current_dir, f"plots/{title} - {stat}.png")
    make_path(plot_path)
    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(stat)
    plt.plot(arrVal, label='Validation ' + stat)
    plt.plot(arrTrain, label='Training ' + stat)
    plt.legend(ncol=4, bbox_to_anchor=(0.5,-0.5), loc='lower center', edgecolor='w')
    plt.tight_layout()
    plt.savefig(plot_path)
