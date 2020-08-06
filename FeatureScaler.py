from sklearn.preprocessing import StandardScaler
import numpy as np

class featureScaler:
    def __init__(self,npDatasetPath):
        self.X = np.load(npDatasetPath)
        
    def scale(self):
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)
        return self
    
#print(featureScaler("players_X.npy").scale().X)