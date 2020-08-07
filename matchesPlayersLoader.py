import pandas as pd
import numpy as np
import os
import playersLoader as pLoader
from dataRecoveryModel import CompDefAwarRecovery
from sklearn.preprocessing import StandardScaler
from playersDfCleaner import PlayersDfCleaner  as pCleaner

matches = pd.read_json("clean_PL_RESULTS.json")

numpyMatches = []

for match in matches.index:
    m = matches.iloc[match,:]
    """
    homeTeamDf = pCleaner(pLoader.getSquadDataFrame(m["home team"],m["date"].timestamp()).head(18),cleaned=1).toML().df
    awayTeamDf = pCleaner(pLoader.getSquadDataFrame(m["away_team"],m["date"].timestamp()).head(18),cleaned=1).toML().df
    """
    try:
        homeTeamDf = pCleaner(pLoader.getSquadDataFrame(m["home team"],m["date"].timestamp()),cleaned=1).dfToBestSquad().toML().df
        awayTeamDf = pCleaner(pLoader.getSquadDataFrame(m["away_team"],m["date"].timestamp()),cleaned=1).dfToBestSquad().toML().df
        homeAwayDf = homeTeamDf.append(awayTeamDf)
        homeAwayDf.drop(columns=["Composure","Defensive Awareness"],inplace=True)
    except:
        curPath = str(os.path.realpath(__file__)).split('\\matchesPlayersLoader.py')[0]+ '\\' 
        os.chdir(curPath)
        np.save("numpyMatchesV3I.npy",np.asarray(numpyMatches))
    try:
        dataRecovery = CompDefAwarRecovery().predict(StandardScaler().fit_transform(homeAwayDf))
    except:
        continue
    
    homeAwayDf["Composure"] = dataRecovery[:,0]
    homeAwayDf["Defensive Awareness"] = dataRecovery[:,1]
    homeAwayDf = homeAwayDf.sort_index(axis=1)
    goalDif = m["home_team_score"] - m["away_team_score"]
    
    if goalDif > 0:
        numpyMatch = np.append(homeAwayDf.to_numpy(),1)
    elif goalDif < 0:
        numpyMatch = np.append(homeAwayDf.to_numpy(),2)
    else:
        numpyMatch = np.append(homeAwayDf.to_numpy(),0)
    
    if numpyMatch.shape[0] == 1065:
        numpyMatches.append(numpyMatch.reshape(homeAwayDf.shape[0] * homeAwayDf.shape[1] + 1))
        print(match)
        print(np.asarray(numpyMatches).shape)
       

curPath = str(os.path.realpath(__file__)).split('\\matchesPlayersLoader.py')[0]+ '\\' 
os.chdir(curPath)
np.save("numpyMatchesV3.npy",np.asarray(numpyMatches))
