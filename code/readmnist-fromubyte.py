#! /bin/python3


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from matplotlib import pyplot as plt


class Mnistfromubyte(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir= root_dir
        self.train= train
        self.transform= transform
        
        # Assemble pathes with os abstraction
        if self.train:
            self.image_path= os.path.join(self.root_dir, 'train-images-idx3-ubyte')
            self.label_path= os.path.join(self.root_dir, 'train-labels-idx1-ubyte')
        else:
            self.image_path= os.path.join(self.root_dir, 't10k-images-idx3-ubyte')
            self.label_path= os.path.join(self.root_dir, 't10k-labels-idx1-ubyte')
        
        # Load data
        self.images= self._read_images()
        self.labels= self._read_labels()


    def _read_images(self):
        with open(self.image_path, 'rb') as f:
            # Skip header bytes
            f.read(16)
            # Read images by file descriptor loop
            buffer= f.read()
            images= np.frombuffer(buffer, dtype=np.uint8)
            # Reshape images
            images= images.reshape(-1, 28, 28)
        return images


    def _read_labels(self):
        with open(self.label_path, 'rb') as f:
            # Skip header bytes
            f.read(8)
            # Read labels by file descriptor loop
            buffer= f.read()
            labels= np.frombuffer(buffer, dtype=np.uint8)
        return labels


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        if self.transform:
            image= self.transform(np.array(self.images[idx])).squeeze()
        else:
            image= torch.tensor(np.array(self.images[idx]))
        label= self.labels[idx]   
        return image, label


# Transform
transform= transforms.Compose( [ 
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
    ])


### Load data

path= "../data/mnistubyte/mnist"

# Custom dataset 
train_data= Mnistfromubyte(root_dir= path, train= True, transform= transform)
test_data= Mnistfromubyte(root_dir= path, train= False, transform= transform)


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

