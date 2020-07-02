import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(4)

#load data
data = pd.read_excel('Molding_Data.xlsx') #using pandas to load the excel file
print(data.shape) #check the number of samples and number of variables
print(data.head(5))
print(np.any(data.isnull() == True))

data = data.dropna(axis=0, how='any') #drop nan from the data

#if there is null N/A entries in the excel file
#print(np.any(data.isnull == True)) #data.isnull will get back the same dimesnion as data
#data = data.dropna(axis = 0, how  = 'any') #
#data.describe()

#specify the prediction and response variables
#1st col = index, 2nd = output, 3rd-6th = inputs 
#X = data.iloc[:,2:6]
X = data.loc[:,['T_Mold','T_Melt','P_Gate','P_Runner']]
Y = data.loc[:,'Part thickness']
X = X.to_numpy()  #covert the data loaded by Pandas to numpy arrays
Y = Y.to_numpy()
#print(X)
#print(Y)
#print(X.shape, Y.shape)

# Now, we have 2 arrays: X = array of the 4 column data, Y = array of part thickness data

# data normalization 

#normalization to gaussian distribution
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler() # (x - u) / s
#X = sc.fit_transform(X)

#normalization to [0 1] range
for i in range(X.shape[1]):
    data_ = X[:,i]
    X[:,i] = (data_-np.amin(data_))/(np.amax(data_)-np.amin(data_))  

const = np.ones((len(X),1)) #this is to create a constant besides the variables

X = np.concatenate((X,const),axis = 1)

# split training and testing data

index = np.arange(len(Y))
np.random.shuffle(index) #disorder the original data

m = np.ceil(0.7*len(Y)) # 70% for training and 30% for testing
m = int(m) #covert float type to int type
X_Train = X[index[0:m],:] #get 70% data for training
Y_Train = Y[index[0:m]]

X_Test = X[index[m:],:] #remaining 30% data for training
Y_Test = Y[index[m:]]

print(X_Train.shape, Y_Train.shape)
print("X_Train: ", X_Train.shape, "\nY_Train:", Y_Train.shape, "\nX_Test:", X_Test.shape, "\nY_Test:", Y_Test.shape)


# define loss function, Mean Square Error (MSE)

def cost_function(X, Y, B):
    J = np.sum((X.dot(B)-Y)**2)/(2*len(Y)) 
    return J
    

# multiple regression algorithm, gradient descent (X,Beta,Y) Y = X*Beta +error
# Beta, loss_history = gradient_descent(X, Y, Beta,alpha,Iterations)
# create iterative gradient descent training
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = np.zeros(iterations)
    for iteration in range(iterations):
        loss = X.dot(B) - Y
        gradient = X.T.dot(loss)/len(Y)
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost 
    
    return B, cost_history
    
    
    
# training
B = np.random.random(X_Train.shape[1]) # Initial Coefficients
alpha = 0.05
iter_ = 5000
newB, cost_history = gradient_descent(X_Train, Y_Train, B, alpha, iter_)
#print(newB)
plt.title("Training Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss History")
plt.plot(np.arange(iter_), cost_history)


# testing 
y_predicted = X_Test.dot(newB)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.scatter(Y_Test, y_predicted)


# performance evaluation
def r2(y_predicted,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_predicted-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)

print("R^2 = ", r2(y_predicted, Y_Test))

#RMSE, root mean sqaure error, e.g. house price is 1M, (1M-RMSE - 1M+RMSE)
RMSE = np.sqrt(np.sum((Y_Test-y_predicted)**2)/len(Y_Test))
print("RMSE = ", RMSE, " mm") 
