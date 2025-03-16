#! /bin/python3


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from matplotlib import pyplot as plt


class Mnistfromjpg(Dataset):
    
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir= root_dir
        self.train= train
        self.transform= transform
        self.data= []
        self.labels= []

        if self.train: 
            subset= 'mnist-training'
        else: 
            subset= 'mnist-testing'
        
        # Assemble pathes with os abstraction
        for digit in range(10):
            digit_dir= os.path.join(self.root_dir, subset, str(digit))
            for img_name in os.listdir(digit_dir):
                img_path= os.path.join(digit_dir, img_name)
                self.data.append(img_path)
                self.labels.append(digit)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path= self.data[idx]
        image= Image.open(img_path).convert('L')
        if self.transform:
            image= self.transform(np.array(image)).squeeze()
        else:
            image= torch.tensor(np.array(image))
        label = self.labels[idx]
        return image, label


# Transform
transform= transforms.Compose( [ 
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
    ])


### Load data

path= "../data/mnistjpg/mnist"

# Custom dataset 
train_data= Mnistfromjpg(root_dir= path, train= True, transform= transform)
test_data= Mnistfromjpg(root_dir= path, train= False, transform= transform)


# Data loaders
train_loader= DataLoader(train_data, batch_size=64, shuffle=True)
test_loader= DataLoader(test_data, batch_size=1)


# Test call
for images, labels in train_loader:
    print("")
    print("") 
    print("tensor shapes")
    print(images.shape, labels.shape)
    print("")
    print("")
    print("true label")
    print(labels[0].numpy())
    print("")
    print("")
    plt.figure(figsize=(6, 6))
    plt.imshow(images[0, :, :])
    plt.show()
    break


# List all objects of the environment 
print("")
print("")
print("List of all objects")     
print(str(dir()))



