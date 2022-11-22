   #### Specify all the paths here #####
traindir = '/data/PublicDataSets/ImageNet-2012/ILSVRC2012/train/'
valdir = '/data/PublicDataSets/ImageNet-2012/ILSVRC2012/val/'

path_to_save_check_points = '/data/home/acw676/ImageNet/weights/'+'/mobilenet_v3_large'
path_to_save_Learning_Curve = '/data/home/acw676/ImageNet/weights/'+'/mobilenet_v3_large'

        #### Specify all the Hyperparameters\image dimenssions here #####
Batch_Size = 256
Max_Epochs = 100
LEARNING_RATE=0.0001
Patience = 5

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
from Early_Stopping import EarlyStopping

### Data_Generators ########
   ### Load the Data using Data generators and paths specified #####
   #######################################
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
            normalize,
        ])
    )
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Batch_Size, shuffle = True,
        num_workers=0, pin_memory=True)
 
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

print(len(train_loader))   ### same here
print(len(val_loader))   ### same here


### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_Acc = []  # all training epochs



import torchvision.models as models
model_ = models.mobilenet_v3_large(pretrained=False)

### Next we have all the funcitons which will be called in the main for training ####
    
### 2- the main training fucntion to update the weights....
def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,scaler):
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    model.train()
    loop = tqdm(loader_train)
    for batch_idx, (image,label) in enumerate(loop):
        image = image.to(device=DEVICE)  
        label = label.to(device=DEVICE)
        
        with torch.cuda.amp.autocast():
            out= model(image)   
            loss = loss_fn1(out, label)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses.append(float(loss))
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (image,label) in enumerate(loop):
        image = image.to(device=DEVICE)  
        label = label.to(device=DEVICE)

        with torch.no_grad():
            
            out= model(image)   
            loss = loss_fn1(out, label)
            
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses.append(float(loss))

    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    
### 3 - this function will save the check-points 
def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

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

CE = torch.nn.CrossEntropyLoss()

## 7- This is the main Training function, where we will call all previous functions
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=Patience, verbose=True)

def main():
    model = model_.to(device=DEVICE,dtype=torch.float)
    loss_fn =CE
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9),lr=LEARNING_RATE)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    # optimizer = optim.SGD(model.parameters(), momentum=0.9 ,lr=LEARNING_RATE)  ### SGD
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        train_loss,valid_loss = train_fn(train_loader,val_loader, model, optimizer, loss_fn,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        Acc_= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_Acc.append(Acc_.detach().cpu().numpy())
        
        early_stopping(valid_loss, Acc_)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()

### This part of the code will generate the learning curve ......

avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_Acc=avg_valid_Acc
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_Acc)+1),avg_valid_Acc,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
