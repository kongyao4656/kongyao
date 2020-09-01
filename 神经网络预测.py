import pandas as pd
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

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
testY = Label[trainNum:]

#模型构建
model = Sequential()
model.add(Dense(units=64,input_dim=trainX.shape[1],activation='sigmoid'))
model.add(Dense(units=256,activation='sigmoid'))
model.add(Dense(units=1,activation='linear'))
optim = keras.optimizers.Adam(lr=0.00001)
model.compile(optimizer=optim, loss=keras.losses.BinaryCrossentropy(), metrics=['mae','acc'])
model.summary()
DEN = model.fit(trainX,trainY, batch_size=64,epochs=20, validation_data=(testX, testY))

#模型可视化
plt.figure()
plt.plot(DEN.history['loss'], label='train')
plt.plot(DEN.history['val_loss'],label='val_loss')
plt.legend()
plt.show()





