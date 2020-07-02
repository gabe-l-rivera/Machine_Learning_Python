#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

np.random.seed(2)
torch.random.manual_seed(2)


# In[51]:


#load data
data = pd.read_excel('Molding_Data.xlsx') #from panda
print(data.shape)
print(np.any(data.isnull == True)) #data.isnull will get back the same dimesnion as data
data = data.dropna(axis = 0, how  = 'any') #
#data.describe()
data.head()
#data['Part thickness'][:5]


# In[52]:


# X = data.loc[:,['sqft_living','bathrooms','sqft_above']] #Variables, or using data.iloc[:,2:6]
X = data.loc[:1000,['T_Mold', 'T_Melt','P_Gate', 'P_Runner']]
Y = data.loc[:1000,'Part thickness'] # Target

X = X.to_numpy() #convert data frame to numpy array
Y = Y.to_numpy()

print(X.shape, Y.shape)


# In[53]:


#normalization to [0 1] range 
X_Norm = np.zeros((X.shape[0],X.shape[1]))
for i in range(X.shape[1]):
    data_ = X[:,i]
    X_Norm[:,i] = (data_-np.amin(data_))/(np.amax(data_)-np.amin(data_)) 

Y_Max = np.amax(Y)
Y_Min = np.amin(Y)
Y_Norm = (Y-Y_Min)/(Y_Max-Y_Min) #normalization of Y. denormalization: Yprediction =  Yprediction*(Y_Max-Y_Min)+Y_Min


# In[54]:


# split training and testing data
index = np.arange(len(Y_Norm))
np.random.shuffle(index) #disorder the original data

m = np.ceil(0.7*len(Y)) # 70% for training and 30% for testing
m = int(m) #covert float type to int type
X_Train = X_Norm[index[0:m],:] #get 70% data for training
Y_Train = Y_Norm[index[0:m]]

X_Test = X_Norm[index[m:],:] #remaining 30% data for training
Y_Test = Y_Norm[index[m:]]

#change Y from 1-dim array to 2-dim arrays, to avoid dimensional discrepancy issue
Y_Train = np.reshape(Y_Train,(len(Y_Train),1)) #(701,)->(701,1)
Y_Test = np.reshape(Y_Test,(len(Y_Test),1)) #(300,)->(300,1)
print(Y_Train.shape, Y_Test.shape)


# In[55]:


#numpy use array as main data structure, torch use tensor as data structure
X_Train_Tensor = torch.tensor(X_Train).float()
Y_Train_Tensor = torch.tensor(Y_Train).float()
X_Test_Tensor = torch.tensor(X_Test).float()
Y_Test_Tensor = torch.tensor(Y_Test).float()


# In[56]:


#define a neural network class, based on torch.nn, inherit from a exsiting class
class NeuralNetwork(torch.nn.Module):
    def __init__(self,layer_numbers):
        super().__init__() #fixed when inheriting from a exsiting class, super refers to torch.nn.module
        # specify the layers
        self.hidden = torch.nn.Linear(layer_numbers[0], layer_numbers[1])
        self.output = torch.nn.Linear(layer_numbers[1], layer_numbers[2])
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# In[57]:


layer_numbers= [4,10,1]
Epoches  = 2000
Net = NeuralNetwork(layer_numbers) #define an object beloning to the defined class
loss_history = []

criterion =  torch.nn.MSELoss() #define the loss function
optimizer =  torch.optim.Adam(Net.parameters(), lr = 0.03)  
#SGD/Adam, lr: learning rate 0.03 is best from what ive seen in this model

for i in range(Epoches):
    Y_pred = Net(X_Train_Tensor)
    loss = criterion(Y_pred, Y_Train_Tensor)
    #backwards training, fixed for every nerual network defined upon torch 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss)
    
plt.figure()
plt.title("Training Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss History")
plt.plot(np.arange(Epoches),loss_history)


# In[58]:


Y_predict = Net(X_Test_Tensor) 
Y_prediction =  Y_predict.detach()*(Y_Max-Y_Min)+Y_Min
Y_Test = Y_Test*(Y_Max-Y_Min)+Y_Min
print(Y_Test)
plt.figure()
plt.scatter(Y_Test,Y_prediction, c = 'b', marker = 'o')
plt.xlim(Y_Min, Y_Max)
plt.ylim(Y_Min, Y_Max)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot([Y_Min, Y_Max], [Y_Min, Y_Max], 'k')
plt.show()


# In[59]:


def r2(y_predicted,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_predicted-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)

print("R^2 = ", r2(Y_prediction.numpy(), Y_Test))


# In[60]:


RMSE = np.sqrt(np.sum((Y_Test-Y_prediction.numpy())**2)/len(Y_Test))
print("RMSE = ", RMSE, "mm") 

# the avg difference in actual vs. predicted = RMSE mm 


# In[ ]:




