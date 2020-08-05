"""
This script contains 2 functions:
    - nearest : returns the nearest item to the pivot item (used to find the nearest TimeStamp)
    - getSquadDataFrame : returns the DataFrame of the squad which played nearly at a certain time. (TimeStamp param)
"""
import pandas as pd
import glob
import os
import datetime

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def getSquadDataFrame(team,timestamp):
    curPath = str(os.path.realpath(__file__)).split('\\playersLoader.py')[0]+ '\\dispatchedPlayers\\' 
    os.chdir(curPath)
    filePathsArray = glob.glob(curPath + '*')
    dateItems = []
    teamsFileNamesDict = {}
    for path in filePathsArray:
        if team in path:
            teamsFileNamesDict[path.split("\\")[-1]] = path
            for fileName in teamsFileNamesDict.keys():
                dateStr = fileName.split(" ")[-1].split('-')
                dateItems.append(datetime.datetime(int(dateStr[0]),int(dateStr[1]),int(dateStr[2])).timestamp())
    try:
        nearestDate = nearest(dateItems,timestamp)
    except:
        print(team + "Not found in the database") 
    return pd.read_json(teamsFileNamesDict[str(team) + " "+ str(datetime.datetime.fromtimestamp(nearestDate)).split(" ")[0]])



#print(getSquadDataFrame("Arsenal",datetime.datetime.now().timestamp()).head(10))
#print(getSquadDataFrame("Manchester City",datetime.datetime.now().timestamp()).head(10))