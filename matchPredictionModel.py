## Importing libraries
import numpy as np
from playersDfCleaner import PlayersDfCleaner as cleaner  
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

data = np.load("numpyMatches.npy")
data = data.astype('int64')
X = data[:,:-1]
Y = data[:,-1] 

xTrain, xTest, yTrain, yTest = train_test_split(X, Y,test_size = 0.2, random_state = 0)

print("XTrain has shape: " + str(xTrain.shape))
print("XTest has shape: " + str(xTest.shape))
print(xTest.dtype)

## Training a linear SVM classifier 
from sklearn.svm import SVC 
print("Training.....")
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(xTrain, yTrain) 
svm_predictions = svm_model_linear.predict(xTest) 

## Model accuracy for xTest   
accuracy = svm_model_linear.score(xTest, yTest) 
print("Accuracy: "+ str(accuracy))


if __name__ == '__main__':
    pass
