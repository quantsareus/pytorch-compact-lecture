#! /bin/python3


import numpy as np
from numpy.random import rand
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
# from sklearn.preprocessing import LabelEncoder


######################################################################################################################
# 
# Load data 
# 


# "archive.ics.uci.edu/ml/datasets/wine+quality"

# Wine quality data set 
data00= pd.read_csv("../data/wine/winequality-white.csv", delimiter= ";", header= 0)


######################################################################################################################
# 
# Prepare data  
# 


### Clean colnames 

data00_cnames= list(data00.columns)
for i in range(0, len(data00_cnames)):
    data00_cnames[i]= str(data00_cnames[i]).replace(" ", "_")
data00.columns= data00_cnames
print( data00.shape)
print("")


### Feature engineering of the quality score 

## Version B By Pandas

y= pd.get_dummies(data00["quality"], dtype= np.int64)

y_cnames= list(y.columns)
for i in range(0, len(y_cnames)):
    y_cnames[i]= "y_" +str(y_cnames[i])

print( "pandas-dummies are category fill sensitive")
print( y.shape)
print("")
print( y)
print("")


### partition the data into "1"=train, "2"=test, "3"=validate 

rnd= rand(data00.shape[0])
partition= np.zeros(data00.shape[0], dtype= np.int64)
partition[ (0.0 < rnd) & (rnd < 4/7)]= 1
partition[ (4/7 < rnd) & (rnd < 6/7)]= 2
partition[ (6/7 < rnd) & (rnd < 1.0)]= 3
partition= pd.DataFrame(partition)

data= pd.concat( [partition, data00, y], axis="columns" )

data_cnames= ["partition"] +data00_cnames +y_cnames
data.columns= data_cnames

print( data.shape)
print("")
print( data.columns)
print("")
print( data)
print("")


######################################################################################################################
# 
# Save prepared data  
# 


# data.to_csv("prepared-dset.csv")


######################################################################################################################
# 
# Data Loader
# 


class CustomWine(Dataset):

    def __init__(self, dset, datapart="train", norm_x=True, y_out="dummy"):
        self.dset= np.array(dset, dtype= np.float32)
        self.datapart= datapart
        self.norm_x= norm_x
        self.y_out= y_out
        
        if self.datapart == "train":
            self.data= torch.tensor(self.dset[ self.dset[ :, 0]== 1, :])
        elif self.datapart == "test":
            self.data= torch.tensor(self.dset[ self.dset[ :, 0]== 2, :])
        elif self.datapart == "validation":
            self.data= torch.tensor(self.dset[ self.dset[ :, 0]== 3, :])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x= self.data[ :, 1:12]
        if self.norm_x:
            for i in range( 0, 11):
                x[ :, i]= 5* ( x[ :, i] -x[ :, i].mean() )/ ( x[ :, i].max() -x[ :, i].min() ) 
        
        y_dum= self.data[ :, 13:]   
        if self.y_out == "dummy":
            y= y_dum
        elif self.y_out == "logitscore":
            dim= list(y_dum.size())
            y= (-1)* torch.ones(dim)
            y[y_dum == 1]= 1
            y= 10* y
        elif self.y_out == "category":
            y= self.data[ :, 12].unsqueeze(1)
        
        return x, y


# y_out="dummy"
# y_out="logitscore"
y_out="category"

train_data= CustomWine(data, datapart="train", norm_x=True, y_out=y_out)
test_data= CustomWine(data, datapart="test", norm_x=True, y_out=y_out)

"""
# Check trein_data, test_data
x_train0, y_train0= train_data[0]
print("x_train")
print(x_train0)
print("")
print("y_train")
print(y_train0)
print("")
x_test0, y_test0= test_data[0]
print("x_test0")
print(x_test0)
print("")
print("y_test0")
print(y_test0)
print("")
"""

# 1. model

no_in= 11

# no_hidden1= 12 
# no_hidden1= 18  
# no_hidden1= 24 
no_hidden1= 36 

# no_out= 7
no_out= 1


model= nn.Sequential(
    nn.Linear(no_in, no_hidden1), 
    
    # limited value activation functions 
    nn.Sigmoid(), 
    # nn.Tanh(),
    # nn.Hardsigmoid(),
    # nn.Hardtanh(),
    # nn.Softmax(), 
    # nn.LogSoftmax(), 
    ######################################
    # unlimited value activation functions 
    # nn.ReLU(),
    # nn.LeakyReLU(negative_slope= 0.10),
    # nn.Softplus(), 
    nn.Linear(no_hidden1, no_out),
    )


# 2. loss functions

lossfn= nn.MSELoss()

# lossfn= nn.CrossEntropyLoss()

"""
# logcosh loss
def lossfn(y, y_p):
    errors= (y -y_p).cosh().log().mean()
    return errors

# L1 loss 
def lossfn(y, y_p): 
    errors= (abs(y -y_p)**1).mean()
    return errors
"""


# gof functions

"""
# R-L2
def goffn(y, y_p):
    y_mean= y.mean()
    errors= (abs(y -y_p)**2).sum()
    reference= (abs(y- y_mean)**2).sum()
    fit= 1 -( errors/ reference )
    return fit
"""

# R-L1
def goffn(y, y_p):
    y_mean= y.mean()
    errors= (abs(y -y_p)**1).sum()
    reference= (abs(y- y_mean)**1).sum()
    fit= 1 -( errors/ reference )
    return fit

# 3.1 optimization

# optimizer= optim.Adam(model.parameters(), lr= 1e-1) 
# optimizer= optim.Adam(model.parameters(), lr= 5e-2) 
# optimizer= optim.Adam(model.parameters(), lr= 2.5e-2) 
# optimizer= optim.Adam(model.parameters(), lr= 1e-2)  
# optimizer= optim.Adam(model.parameters(), lr= 7.5e-3)  
# optimizer= optim.Adam(model.parameters(), lr= 5e-3) # 
# optimizer= optim.Adam(model.parameters(), lr= 2.5e-3) # log entr 0.317529
# optimizer= optim.Adam(model.parameters(), lr= 1e-3) # log entr 0.346875, # categ 36 0.089974
# optimizer= optim.Adagrad(model.parameters(), lr= 7.5e-4) # dummy mse opt 0.270184, 0.252302 
# optimizer= optim.Adagrad(model.parameters(), lr= 5e-4)  
# optimizer= optim.Adagrad(model.parameters(), lr= 1e-4)  

# optimizer= optim.Adagrad(model.parameters(), lr= 1e-1) 
# optimizer= optim.Adagrad(model.parameters(), lr= 5e-2) 
# optimizer= optim.Adagrad(model.parameters(), lr= 2.5e-2) 
optimizer= optim.Adagrad(model.parameters(), lr= 1e-2) # categ 36 0.128673, 0.126513, 0.115040, 0.125944
# optimizer= optim.Adagrad(model.parameters(), lr= 7.5e-3) # log entr 0.385442, 0.357100, 0.367847, 0.373066
# optimizer= optim.Adagrad(model.parameters(), lr= 5e-3) # dummy mse 0.258967
# optimizer= optim.Adagrad(model.parameters(), lr= 2.5e-3) # dummy mse 0.245514
# optimizer= optim.Adagrad(model.parameters(), lr= 1e-3) # 
# optimizer= optim.Adagrad(model.parameters(), lr= 7.5e-4) 
# optimizer= optim.Adagrad(model.parameters(), lr= 5e-4)   
#optimizer= optim.Adagrad(model.parameters(), lr= 1e-4)  


# 3.2 training loop


def training_loop(epochs_max, epochs_min, testcycle, gain_min, optimizer, model, lossfn):
    
    loss_prev= 1e10
    goftest_prev= -1e10
    for epoch in range(0, epochs_max):
        
        x, y= train_data[:]

        y_pred= model(x)
        
        # Straight order
        loss= lossfn(y, y_pred)
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
               
        if epoch % testcycle == 0:
            
            x_test, y_test= test_data[:]
            
            y_test_pred= model(x_test)
            goftest= goffn(y_test, y_test_pred)
            
            print("Epoch %d, Loss %f, Goftest %f" % (epoch, float(loss), float(goftest)))
            # print(y_pred[ 0, 0, :].detach().numpy())
            
            if (epoch >= epochs_min) & (goftest > 0.0):
                if  ((loss -loss_prev) < 0) & ((goftest -goftest_prev) < gain_min):
                    print("")
                    print("STOPPED !!")
                    break
            
            loss_prev= loss
            goftest_prev= goftest
        

# 4. run
training_loop(
    epochs_max= 10001,
    epochs_min= 20,
    testcycle= 10,
    gain_min= 1e-6,
    optimizer= optimizer,
    model= model,
    lossfn= lossfn, 
    # x= x,
    # y= y,
    )



