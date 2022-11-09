
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import matplotlib.pyplot as plt


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

traindir = r'C:\My_Data\mahmud\val'

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
        train_dataset, batch_size=3, shuffle = True,
        num_workers=0, pin_memory=True)
 
 
a = iter(train_loader)
a1 = next(a)
img = a1[0][2,:,:,:]
img = torch.permute(img, (1,2,0))
plt.figure()
plt.imshow(img)
print(a1[1])
