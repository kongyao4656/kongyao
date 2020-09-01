import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt

dta = pd.read_csv('./data/car.txt',header=None, names=['buying','maint','doors','persons','lug_boot','safty','classValues'])
print(dta)
#独热编码
data = pd.get_dummies(dta,columns=['buying','maint','doors','persons','lug_boot','safty','classValues'])
print(data)

#数据划分
dataSet = data.values
trainNum = int(0.8 * len(dataSet))
trainX = dataSet[:trainNum,:-4]
trainY = dataSet[:trainNum,-4:]

testX = dataSet[trainNum:, :-4]
testY = dataSet[trainNum:, -4:]

#模型构建
model = Sequential()
model.add(Dense(units=16, input_dim=trainX.shape[1], activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=32,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation='softmax'))
optim = keras.optimizers.Adam(lr=0.001)
model.compile(optim,loss=keras.losses.CategoricalCrossentropy(),metrics=['mae','acc'])
model.summary()
result = model.fit(trainX,trainY,batch_size=32,epochs=40,validation_data=(testX,testY))

#display
plt.figure()
plt.plot(result.history['loss'], label='train_loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

