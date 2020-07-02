#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(2)


# In[111]:


#load data
data = pd.read_excel('AAPL_Data.xlsx')
print(data.shape)
print(np.any(data.isnull == True)) #data.isnull will get back the same dimesnion as data
data = data.dropna(axis = 0, how  = 'any')
#data.describe()
data.head()
#data['Part thickness'][:5]


# In[112]:


# X = data.loc[:,['sqft_living','bathrooms','sqft_above']] #Variables, or using data.iloc[:,2:6]
X = data.loc[:1000,['High', 'Low','Open', 'Volume']]
Y = data.loc[:1000,'Close'] # Target

X = X.to_numpy() #convert data frame to numpy array
Y = Y.to_numpy()

print(X.shape, Y.shape)


# In[113]:


#normalization to [0 1] range 
X_Norm = np.zeros((X.shape[0],X.shape[1]))
for i in range(X.shape[1]):
    data_ = X[:,i]
    X_Norm[:,i] = (data_-np.amin(data_))/(np.amax(data_)-np.amin(data_)) 

Y_Max = np.amax(Y)
Y_Min = np.amin(Y)
Y_Norm = (Y-Y_Min)/(Y_Max-Y_Min) #normalization of Y. denormalization: Yprediction =  Yprediction*(Y_Max-Y_Min)+Y_Min


# In[114]:


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


# In[115]:


# define sigmoid and sigmoid derivative
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_derivative(output):
    return output*(1-output)


# In[116]:


# 3-layer perceptron: weight initialization, forward, backwards
class NeuralNetwork():#contains multiple variables and functions, layer_numbers: [3,10,1]
    def __init__(self,X,Y,layer_numbers,learning_rate,Epoches): #in parenthesis, list all input variables to define the class
        self.X = X #convert a local variable to a global variable
        self.Y = Y
        self.learning_rate = learning_rate
        self.Epoches = Epoches
        # initialize network weights
        self.W0 = np.random.rand(layer_numbers[0],layer_numbers[1])
        self.W1 = np.random.rand(layer_numbers[1],layer_numbers[2])
        
        self.error_history = []
        self.epoch = []
    
    def forward(self):
        self.hidden_output = sigmoid(np.dot(self.X, self.W0))
        self.output = sigmoid(np.dot(self.hidden_output, self.W1))
    
    def backwards(self):
        #loss function, prediction error
        self.error = np.average(np.abs(self.output-self.Y)) #sum(|prediction-actual|)/No.(data)
        W1_gradient = np.dot(self.hidden_output.T,(self.output-self.Y)*sigmoid_derivative(self.output)) #W1 gradient
        hidden_error = np.dot((self.output-self.Y)*sigmoid_derivative(self.output), self.W1.T) #error of hidden layer, dJ/dH
        W0_gradient = np.dot(self.X.T, hidden_error*sigmoid_derivative(self.hidden_output)) #W0 gradient
        
        self.W1= self.W1 - self.learning_rate*W1_gradient
        self.W0= self.W0 - self.learning_rate*W0_gradient
    
    def train(self):
        for i in range(self.Epoches):
            self.forward()
            self.backwards()
            self.epoch.append(i)
            self.error_history.append(self.error)
            #for the purpose of plotting the traing curve, make a record of prediction error for every epoch
            
    def prediction(self, new_data): #predict the test output based on test X
        hidden_output = sigmoid(np.dot(new_data, self.W0))
        output = sigmoid(np.dot(hidden_output, self.W1))
        return output
        


# In[117]:


layer_numbers = [4,10,1] #change middle index to alter # of layers
learning_rate  = 0.03
#learning rate 0.3 is best from what ive seen in this model
Epoches = 5000
Net = NeuralNetwork(X_Train,Y_Train,layer_numbers,learning_rate,Epoches) #define a object beloning to the calss
#train the network with the trianing data
Net.train()
plt.figure()
plt.plot(Net.epoch,Net.error_history)
plt.title("Training Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss History")
plt.show()


# In[118]:


#Test the performance
Y_predict = Net.prediction(X_Test) 
Y_prediction =  Y_predict*(Y_Max-Y_Min)+Y_Min
#print(Y_prediction.shape)
Y_Test = Y_Test*(Y_Max-Y_Min)+Y_Min
plt.scatter(Y_Test,Y_prediction, c = 'b', marker = 'o')
plt.xlim(Y_Min, Y_Max)
plt.ylim(Y_Min, Y_Max)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot([Y_Min, Y_Max], [Y_Min, Y_Max], 'k')


# In[119]:


def r2(y_predicted,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_predicted-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)

print("R^2 = ", r2(Y_prediction, Y_Test))


# In[120]:


#RMSE, root mean sqaure error, e.g. house price is 1M, (1M-RMSE - 1M+RMSE)
#print(Y_Test)
#print(Y_predict)

RMSE = np.sqrt(np.sum((Y_Test-Y_prediction)**2)/len(Y_Test))
print("RMSE = ", RMSE) 

# the avg difference in actual vs. predicted = RMSE mm 


# In[ ]:




