#! /bin/python3


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
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
model= nn.Linear(1, 1)


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
        
        y_pred= model(x) # <1>
        
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


# 5. show the results
print("")
print(model.bias)
print("")
print(model.weight)


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


