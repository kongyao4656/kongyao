import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dta = pd.read_csv('../data/car.txt',header=None, names=['buying','maint','doors','persons','lug_boot','safty','classValues'])
print(dta)
#独热编码
data = pd.get_dummies(dta,columns=['buying','maint','doors','persons','lug_boot','safty','classValues'])
print(data)

#数据划分
dataSet = data.values
Features = dataSet[:, :-4]
targets = dataSet[:,-4:]
# trainNum = int(0.8 * len(dataSet))
# # trainX = dataSet[:trainNum,:-4]
# # trainY = dataSet[:trainNum,-4:]
# #
# # testX = dataSet[trainNum:, :-4]
# # testY = dataSet[trainNum:, -4:]
trainx, testx, trainy, testy = train_test_split(Features, targets, test_size=0.2)



cls = RandomForestClassifier(n_estimators=50)
cls.fit(trainx, trainy)
print(cls.score(testx,testy))
