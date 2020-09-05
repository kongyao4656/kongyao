'''
np实现对数几率回归，总结要点：
1 参数w的初始化。主要使用np的random函数进行随机初始化，一个要点是 参数的数量应该是样本特征的数量
2 y的预测值的产生。在对数几率回归中，直接使用公式进行计算即可。主要要有公式。
3 随机梯度下降更新参数。这个就要看公式推导，损失函数对参数的导数，然后用随机梯度下降更新，如果每一个样本一次更新，那么m就是1，如果
一个batchsize更新一次参数，那么m的值就是batchsize。m是指示加累加多少个y和y预测值差值。
4 再缩放的实现。看周志华西瓜书P67 样本不均衡问题下，对分类器的优化。
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LogisticRess():
    def __init__(self, epoch, lr, loss_value):
        self.epoch = epoch
        self.lr = lr
        self.loss_value = loss_value

    def fit(self, X, Y):
        n_sample, n_feature = X.shape
        rnd_val = 1/np.sqrt(n_feature)
        rng = np.random.default_rng()
        self.w = rng.uniform(-rnd_val, rnd_val, size=n_feature)
        self.b = 0
        self.mp = 0
        self.mn = 0
        for j in Y:
            if j == 1:
                self.mp += 1
            else:
                self.mn += 1
        self.resacleValue = self.mp / self.mn

        num_epoch = 0
        prev_loss = 0
        while True:
            cur_loss = 0
            for i in range(n_sample): # 这里设置的是一个样本更新一次参数，可以之后设置为一个batchsize设置一次，在外面再加一层循环即可。
                y_pred = self.sigmoid(X[i])
                y_diff = Y[i] - y_pred
                self.w += self.lr * y_diff * X[i]
                self.b += self.lr * y_diff
                cur_loss += abs(y_diff)
            num_epoch += 1
            loss_diff = abs(cur_loss - prev_loss)
            prev_loss = cur_loss

            if num_epoch >= self.epoch or loss_diff < self.loss_value:
                break

    def sigmoid(self, x):
        return 1/(1 + np.exp(-(np.dot(self.w, x) + self.b)))

    def predict(self, x):
        resultArray = []
        for i in range(x.shape[0]):
            logit = np.dot(self.w, x[i]) + self.b
            resultArray.append(1 if logit > np.log(self.resacleValue) else 0) # 这里使用了再缩放的技术，使得预测结果更加符合样本分布，见周志华西瓜书P67
        return np.array(resultArray)
    def score(self, x, y):
        y_pred = self.predict(x)
        diff = y_pred - y
        bingo = 0
        bad = 0
        for i in diff:
            if i == 0:
                bingo += 1
            else:
                bad += 1
        print('预测准确率为：{}'.format(bingo/len(diff)))
if __name__ == '__main__':
    dta = pd.read_csv('../data/ranked_10min.csv')
    colulist = dta.columns.tolist()
    print(colulist)

    dta['GoldMinDiff'] = dta['blueGoldPerMin'] - dta['redGoldPerMin']
    dta['CSDiff'] = dta['blueCSPerMin'] - dta['redCSPerMin']
    dta.drop(columns=['gameId', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueTotalGold', 'blueTotalExperience',
                      'redWardsPlaced', 'redWardsDestroyed', 'redKills', 'redDeaths', 'redTotalGold',
                      'redTotalExperience', 'redGoldDiff', 'redExperienceDiff', 'blueGoldPerMin', 'redGoldPerMin',
                      'blueCSPerMin', 'redCSPerMin'], inplace=True)

    # 划分训练集和测试集
    Label = dta['blueWins'].values
    dta.drop(columns=['blueWins'], inplace=True)
    Features = dta.values

    trainx, testx, trainy, testy = train_test_split(Features, Label, test_size=0.2)
    epoch = 10
    lr = 0.001
    loss_value = 0.00001
    LogisticCls = LogisticRess(epoch, lr, loss_value)
    LogisticCls.fit(trainx, trainy)
    LogisticCls.score(testx,testy)

