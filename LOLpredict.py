import pandas as pd
import numpy as np
from sklearn.svm import SVC

dta = pd.read_csv('./data/ranked_10min.csv')

colulist = dta.columns.tolist()
print(colulist)

dta['GoldMinDiff'] = dta['blueGoldPerMin'] - dta['redGoldPerMin']
dta['CSDiff'] = dta['blueCSPerMin'] - dta['redCSPerMin']
dta.drop(columns=['blueWardsPlaced', 'blueWardsDestroyed','blueTotalGold','blueTotalExperience','redWardsPlaced', 'redWardsDestroyed','redKills', 'redDeaths','redTotalGold',
                  'redTotalExperience','redGoldDiff', 'redExperienceDiff','blueGoldPerMin','redGoldPerMin','blueCSPerMin','redCSPerMin'],inplace=True)

#划分训练集和测试集
Label = dta['blueWins'].values
dta.drop(columns=['blueWins'],inplace=True)
Features = dta.values

trainNum = int(0.8 * len(dta))
trainX = Features[:trainNum, :]
trainY = Label[:trainNum]

testX = Features[trainNum:,:]
testY = Label[:trainNum]

#模型构建
model = SVC(kernel='linear')
model.fit(trainX, trainY)
predy = model.predict(testX)
Diff = predy - testY
print(Diff)




