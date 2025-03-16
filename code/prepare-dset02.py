#! /bin/python3


import numpy as np
from numpy.random import rand
import pandas as pd
import torch
# from sklearn.preprocessing import LabelEncoder


######################################################################################################################
# 
# Load data 
# 


# Wine quality data set 
# "archive.ics.uci.edu/ml/datasets/wine+quality"

data00= pd.read_csv("../data/wine/winequality-white.csv", delimiter= ";", header= 0)


#####################################################################################################################
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

### Feature engineering of the quality score 

"""
## Version A By Torch

# Casting to tensor
y_hot= data00["quality"]
y_hot_t= torch.from_numpy(np.array(y_hot))

# Dummy Coding 
y_t = torch.zeros(y_hot_t.shape[0], 11)
y_t.scatter_(1, y_hot_t.unsqueeze(1), 1.0)
print( "torch-dummies are category fill insensitive")
print(y_t.shape)
print("")
"""

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


####################################################################################################################### 
# Save prepared data  
# 


# data.to_csv("prepared-dset.csv")



