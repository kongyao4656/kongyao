import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


dta = pd.read_csv('../data/ranked_10min.csv')
colulist = dta.columns.tolist()
print(colulist)

dta['GoldMinDiff'] = dta['blueGoldPerMin'] - dta['redGoldPerMin']
dta['CSDiff'] = dta['blueCSPerMin'] - dta['redCSPerMin']
dta.drop(columns=['gameId', 'blueWardsPlaced', 'blueWardsDestroyed','blueTotalGold','blueTotalExperience','redWardsPlaced', 'redWardsDestroyed','redKills', 'redDeaths','redTotalGold',
                  'redTotalExperience','redGoldDiff', 'redExperienceDiff','blueGoldPerMin','redGoldPerMin','blueCSPerMin','redCSPerMin'],inplace=True)

#划分训练集和测试集
Label = dta['blueWins'].values
dta.drop(columns=['blueWins'],inplace=True)
Features = dta.values

trainx,testx, trainy,testy = train_test_split(Features, Label, test_size=0.2)

cls = RandomForestClassifier()
cls.fit(trainx,trainy)

print(cls.score(testx,testy))


