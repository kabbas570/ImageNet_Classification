   #### Specify all the paths here #####

valdir = '/data/PublicDataSets/ImageNet-2012/ILSVRC2012/val/'


        #### Specify all the Hyperparameters\image dimenssions here #####
Batch_Size = 256
LEARNING_RATE=0

        #### Import All libraies used for training  #####
import torch    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


### Data_Generators ########
   ### Load the Data using Data generators and paths specified #####
   #######################################
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize([224,224]),
           transforms.ToTensor(),
            normalize,
        ])
    )
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Batch_Size, shuffle = True,
        num_workers=0, pin_memory=True)

print(len(val_loader))   ### same here


### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_Acc = []  # all training epochs



import torchvision.models as models
model_ = models.mobilenet_v3_large(pretrained=True)

def accuracy(output, target, topk=(5,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = (pred == target.view(1, -1).expand_as(pred))
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def check_accuracy(loader, model, device=DEVICE):
    Accuracy = 0 
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(loop):
            data = data.to(device=DEVICE)
            label = label.to(device=DEVICE)
            predictions = model(data)
            Acc_ = accuracy(predictions,label)
            Accuracy = Accuracy+Acc_[0]
    print('\n accuray is : \n', Accuracy/len(loader))
    return Accuracy/len(loader)

def eval_():
    model = model_.to(device=DEVICE,dtype=torch.float)
    check_accuracy(val_loader, model, device=DEVICE)
 
if __name__ == "__main__":
    eval_()
