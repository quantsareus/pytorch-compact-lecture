#! /bin/python3


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# from collections import OrderedDict
from matplotlib import pyplot as plt


# 0. create the data
x= torch.tensor([-4.0, 0.5, 3.0, 6.0, 8.0, 11.0, 13.0, 14.0, 15.0, 21.0, 28.0])
y= torch.tensor([21.8, 35.7, 33.9, 48.4, 48.9, 56.3, 60.4, 55.9, 58.2, 68.4, 81.9])

# Transform y into interval [0, 1]
y= 0.01* y

# Add an extra dimension of size 1
x= x.unsqueeze(1) 
y= y.unsqueeze(1)


# 1. model
no_neurons= 20
model= nn.Sequential(
    
    nn.Linear(1, no_neurons),
    
    ######################################
    # limited activation functions 
    nn.Sigmoid(),
    # nn.Tanh(),
    # nn.Hardsigmoid(),
    # nn.Hardtanh(),
    # nn.Softmax(),
    ######################################
    # unlimited activation functions 
    # nn.ReLU(),
    # nn.LeakyReLU(negative_slope= 0.10),
    # nn.Softplus(),
    
    nn.Linear(no_neurons, 1),    
    )


# 2. loss functions

# auto loss function
lossfn= nn.MSELoss() 


# 3.1 optimization 
# optimizer= optim.Adam(model.parameters(), lr= 1e-3)
# optimizer= optim.Adagrad(model.parameters(), lr= 5e-3)
optimizer= optim.SGD(model.parameters(), lr=5e-3)
# optimizer= optim.RMSprop(model.parameters(), lr= 1e-5)
# optimizer= optim.Adadelta(model.parameters(), lr= 5e-3)


# 3.2 training loop  
def training_loop(n_epochs, optimizer, model, lossfn, x, y):
       
    for epoch in range(0, n_epochs):
        
        y_pred= model(x)
        
        loss= lossfn(y, y_pred)
       
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        print('Epoch %d, Loss %f' % (epoch, float(loss)))


# 4. run
training_loop(
    n_epochs= 10000, 
    optimizer= optimizer,
    model= model,
    lossfn= lossfn,  
    x= x,
    y= y,
    )


# 5. show results
x_= x.squeeze().numpy()
y_= y.squeeze().numpy() / 0.01
pred= model(x).detach().squeeze().numpy() / 0.01

print("")
fig= plt.figure(dpi=600)
plt.plot(x_, y_, "bo")
plt.plot(x_, pred, "rx-", linewidth= 0.50)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 6.save the model
"""
# Load a model containing changes in its structure from a dict
modelstate_dict= torch.load("/tmp/model0.20139074")
model.load_state_dict(modelstate_dict)

# Save a model containing changes in its structure to a dict
path= "/tmp/model"
print("Current save path is: ", path)
print("")

shouldsave= input("Save the nodel? ( y/n + ENTER )")

if shouldsave== "y":    
    torch.save(model.state_dict(), path)
"""
    

