import pandas as pd
from playersDfCleaner import PlayersDfCleaner
import os 

allPlayers = pd.read_json("PL_PLYRS.json")
os.chdir(str(os.path.realpath(__file__)).split('\\playersDispatcher.py')[0]+ '\\dispatchedPlayers\\' )

for team in pd.unique(allPlayers['team']):
    singleTeamPlayers = allPlayers[allPlayers.team == team]
    #singleTeamPlayers = cleanPlayersDf(singleTeamPlayers) # All versions , One team
    for date in pd.unique(singleTeamPlayers['date']):
        singleSquad = PlayersDfCleaner(singleTeamPlayers[singleTeamPlayers.date == date]).clean().df
        singleSquad.to_json(str(team) + " " + str(date).split('T')[0],orient="records")
        print(str(team) + " " + str(date).split('T')[0])
