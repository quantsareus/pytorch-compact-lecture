#! /bin/python3


import numpy as np
import torch
from matplotlib import pyplot as plt


# 0. Create the data
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


# 3.1 optimization
# autogradient


# 3.2 define the autograd training loop
def training_loop(n_epochs, learning_rate, params, x, y):
    for epoch in range(0, n_epochs):       
        
        # Zero params.grad
        if params.grad is not None:
            params.grad.zero_()
        
        b0, b1= params
        
        # Forward propagation
        y_pred= lin_model(x, b0, b1)
        
        # Loss computation
        loss= lossfn(y, y_pred)
        
        # Set autogradient backpropagation point
        loss.backward()
        
        # Disable the autogradient calculation for params update
        with torch.no_grad():
            # In-place update
            params-= learning_rate *params.grad
        
        print('Epoch %d, Loss %f' % (epoch, float(loss)))           
    return params


# 4. run the training loop
params_= training_loop(
    n_epochs= 10000, 
    learning_rate= 1e-3,
    params= torch.tensor([0.0, 0.0], requires_grad=True),
    x= x,
    y= y)


# 5. show the results
print("")
print(params_)

pred=lin_model(x,params_[0],params_[1]).detach()

print("")
fig = plt.figure(dpi=600)
plt.plot(x.numpy(), y.numpy(), "bo")
plt.plot(x.numpy(), pred.numpy(), "rx-", linewidth= 0.50)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


