#! /bin/python3


import numpy as np
import torch
from matplotlib import pyplot as plt


# 0. create the data
x= torch.tensor([-4.0, 0.5, 3.0, 6.0, 8.0, 11.0, 13.0, 14.0, 15.0, 21.0, 28.0])
y= torch.tensor([21.8, 35.7, 33.9, 48.4, 48.9, 56.3, 60.4, 55.9, 58.2, 68.4, 81.9])


# 1. define the prediction model
def lin_model(x, b0, b1):
    y= b0+ b1* x
    return y


# 2. define the loss function
def lossfn(y, y_p):
    errors= ((y- y_p)**2).mean()
    return errors


# autogradient example

# initialization of params; setting requires_grad to True
params= torch.tensor([0.0, 1.0], requires_grad=True) 

# Forward propagation
y_pred=lin_model(x, params[0], params[1])

# Loss computation
loss=lossfn(y, y_pred)

# Call .backward() for autograd
loss.backward()

# display gradient values, dloss/db0, dloss/db1
print("")
print(params.grad)

# Should output 
# tensor([-0.6954, -8.5768])


