'''
多元线性回归。但是还没有写完，因为多维空间下如何表示出数据是一个问题。
另外，回归不同于分类算法，分类算法可以通过与真实值相比较来得出精度，但是回归问题没有一个评估精度的方法。除了MAE,MAPE等等。
'''

import numpy as np


class LinearRess():
    def __init__(self, epoch, lr):
        self.epoch = epoch
        self.lr = lr

    def fit(self,X,Y):
        X = np.array(X)
        X = np.c_[np.ones(len(X)),X] # 增加一个全为1的列向量
        n_sample, n_feature = X.shape
        rnd = 1/np.sort(n_feature)
        rng = np.random.default_rng()
        self.w = rng.uniform(-rnd, rnd, size=n_feature)

        num_epoch = 0
        pre_loss = 0
        for i in range(self.epoch):
            cur_loss = 0
            for j in range(n_sample):
                y_pred = self.compute(X[j])
                y_Diff = (y_pred - Y[j])
                self.w -= self.lr * y_Diff * X[j]

    def predict(self,X):
        X = np.array(X)
        X = np.c_[np.ones(len(X)), X]
        resultArray = []
        for i in range(len(X)):
            resultArray.append(self.compute(X[i]))
        return np.array(resultArray)

    def compute(self, x):
        return np.dot(self.w, x)

if __name__ == '__main__':
    cls = LinearRess(epoch=20,lr=0.001)
    cls.fit()




