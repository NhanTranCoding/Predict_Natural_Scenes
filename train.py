import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.tensorboard import SummaryWriter
from create_dataset import NaturalDataset
from model import CNNForNaturalDataset
import os
import shutil
import numpy as np
from utils import plot_confusion_matrix
from tqdm.autonotebook import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a model CNN")
    parser.add_argument("--data-path", "-d", type=str, default="data")
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--log-path", "-l", type=str, default="tensorboard")
    parser.add_argument("--save-path", "-s", type=str, default="trained_model")
    parser.add_argument("--checkpoint", type=str, default="None")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    start_epoch = 0
    num_epochs = args.epochs
    batch_size = args.batch_size

    # if os.path.isdir(args.log_path):
    #     shutil.rmtree(args.log_path)
    # os.mkdir(args.log_path)
    # if os.path.isdir(args.save_path):
    #     shutil.rmtree(args.save_path)
    # os.mkdir(args.save_path)
    writer = SummaryWriter(args.log_path)

    transform = Compose([
        Resize((128, 128)),
        ToTensor()
    ])
    train_data = NaturalDataset(root=args.data_path, train=True, transform=transform)
    valid_data = NaturalDataset(root=args.data_path, train=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    model = CNNForNaturalDataset()
    criterion = nn.CrossEntropyLoss()
    optimize = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_loss = 1000
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, num_epochs):
        train_loss = []
        valid_loss = []
        model.train()
        progressbar = tqdm(train_dataloader, colour="green")
        for i, (images, labels) in enumerate(progressbar):
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            train_loss.append(loss_value.item())

            optimize.zero_grad()
            loss_value.backward()

            optimize.step()
            writer.add_scalar("Train/Loss", np.mean(train_loss), i + epoch * len(train_dataloader))
            progressbar.set_description("Epoch: {} Iteration: {}/{} Loss: {}".format(epoch + 1, i + 1, len(train_dataloader), np.mean(train_loss)))
        model.eval()
        with torch.no_grad():
            all_predicted = []
            all_labels = []
            for i, (images, labels) in enumerate(valid_dataloader):
                outputs = model(images)
                loss_value = criterion(outputs, labels)
                valid_loss.append(loss_value.item())
                result = torch.argmax(outputs, dim=1)
                all_predicted.extend(result.tolist())
                all_labels.extend(labels.tolist())
            conf = confusion_matrix(all_labels, all_predicted)
            acc = accuracy_score(all_labels, all_predicted)
            print("Epoch {} Accuracy {}".format(epoch, acc))
            writer.add_scalar("Valid/Loss", np.mean(valid_loss), epoch)
            print("Coff: ", conf)
            # plot_confusion_matrix(writer, conf, [i for i in range(6)], epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimize": optimize.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.log_path, "last.pt"))
        if best_loss > np.mean(valid_loss):
            best_loss = np.mean(valid_loss)
            torch.save(checkpoint, os.path.join(args.log_path, "best.pt"))