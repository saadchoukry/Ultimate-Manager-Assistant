import pandas as pd


class PlayersDfCleaner():
    """
    This class contains methods that we will be using to: 
    -   Clean DataFrames
    -   Transform them to Machine Learning uses
    """
    # Dictionary containing (field positions <=> labels) mapping 
    positionMapDict = {'GK':1,'RB':2,'LB':3,'CB':4,'CDM':5,'CM':6,
                   'RW':7,'CF':8,'ST':9,'CAM':10,'LW':11,
                   'LWB':12,'RWB':13,'RM':14,'LM':15,'RM':16,
                   'LF':17,'RF':18}
    
    def __init__(self,df,cleaned = 0, mL = 0):          # Df should be a DataFram freshly read (json)
        self.df = df
        self.cleaned = cleaned                          #  1 if the dataFrame is cleaned 
        self.mL = mL                                    #  1 if the dataFrame is in Machine Learning Format
        
    
    def clean(self):
        #self.df.rename(columns={"Marking":"Defensive Awareness"},inplace=True)
        dfCopy = self.df.copy(deep=True)
        ids = [int(itm[0]) for itm in dfCopy["id"].str.findall('ID: (\d*)') if len(itm)>0]
        names = [itm[0] for itm in dfCopy["id"].str.findall('[\w ]+') if len(itm)>0]
        ages = [int(itm[0]) for itm in dfCopy["gen_info"].str.findall('(\d+)y.o.') if len(itm)>0]
        heights = [int((float(itm[0][0]) * 12 + float(itm[0][2:])) * 2.54)  for itm in dfCopy["gen_info"].str.findall('(\d\'\d+)') if len(itm)>0]
        weights = [int(int(itm[0])* 0.45) for itm in dfCopy["gen_info"].str.findall('(\d+)lbs') if len(itm)>0]
        dfCopy["Player's id"]= ids
        dfCopy["Name"] = names 
        dfCopy["Age"] = ages
        dfCopy["Weight"] = weights
        dfCopy["Height"] = heights
        dfCopy.drop(columns=["id","gen_info"],inplace=True)
        dfCopy.rename(columns={"country":"Country","date":"Date","pos":"Pos","team":"Team"},inplace=True)
        dfCopy = dfCopy.sort_index(axis=1)
        self.df = dfCopy
        self.cleaned = 1
        return self


    def getUsefulColumns(self):
        toDrop = ["Player's id","Marking"]
        for column in self.df.columns:
            if not (self.df[column].dtype == "int64" or  
                    self.df[column].dtype == "float64" or 
                    column == 'Pos'):
                toDrop.append(column)
        self.df.drop(columns=toDrop,inplace=True)
        return self
    
    def toML(self):
        if self.cleaned == 0:
            self.clean()
        elif self.mL == 1:
            return self
        self = self.getUsefulColumns()
        self.df['Pos'].replace(self.positionMapDict,inplace=True)
        return self

"""
df = pd.read_json("PL_PLYRS.json")
exemple = PlayersDfCleaner(df).toML().df
print(exemple.head())
print(exemple.isna().sum())

df = pd.read_json("PL_PLYRS.json")
exemple = PlayersDfCleaner(df).toML().df

print(exemple.isna().sum())
"""