## Importing libraries
from playersDfCleaner import PlayersDfCleaner as cleaner  
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

"""
This class will be used in "matchesLoader.py" in order to recover examples
that are lacking "Composure"/"Defensive Awareness" attributes.
"""

class CompDefAwarRecovery():
    def __init__(self):
        self.model = pickle.load(open("C:\\Users\\Saad\\Desktop\\Ultimate-Coach-Assistant\\ComAndDefAwarModel.sav",'rb'))
    
    def predict(self,dfX):
        return self.model.predict(dfX)
        


if __name__ == '__main__':
    """
    This script contains a sequence of commands used to train the model that will help 
    us recover rows of the dataset (rows lacking 'Composure' and 'Defensive Awareness' 
    attributes)
    """
    
    noNanDf = cleaner(pd.read_json("PL_PLYRS.json")).toML().df.dropna(axis=0)
    dfY = noNanDf[["Composure","Defensive Awareness"]]
    dfX =StandardScaler().fit_transform(noNanDf.drop(columns=["Composure","Defensive Awareness"]))
    xTrain, xTest, yTrain, yTest = train_test_split(dfX, dfY,
                                                    test_size = 0.2, random_state = 0)


    ## Creating a model instance and fitting it to the training set
    model = LinearRegression()
    model.fit(xTrain, yTrain)

    ## Calculating the accuracy of the model [Based on the test set]
    accuracy = model.score(xTest,yTest) #Accuracy = 0.9219
    print("Accuracy: "+ str(accuracy))
    ## Saving the model (parameters mainly) as a file
    pickle.dump(model,open("ComAndDefAwarModel.sav",'wb'))

