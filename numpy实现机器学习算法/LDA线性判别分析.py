import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt

class LDA():
    def __init__(self):
        pass
    def fit(self, X,Y):
        n_feature = X.shape[1]
        rnd = 1/np.sqrt(n_feature)
        rng = np.random.default_rng()
        self.w = rng.uniform(-rnd, rnd, size=n_feature)
        X1 = np.array([X[i] for i in range(len(X)) if Y[i] == 1])
        X2 = np.array([X[j] for j in range(len(X)) if Y[j] == 0])
        #print(X1)

        miu1 = np.mean(X1, axis=0)
        miu2 = np.mean(X2, axis=0)


        conv1 = np.dot((X1 - miu1).T, (X1 - miu1))
        print(conv1)

        conv2 = np.dot((X2 - miu2).T, (X2 - miu2))
        #print(conv1)
        #print(conv2)
        Sw = conv1 + conv2
        print(Sw.shape)
        #print(Sw)
        print(np.mat(Sw).I)
        self.w  = np.dot(np.mat(Sw).I, miu1 - miu2)

        x1_new = np.dot(X1, self.w.T)
        x2_new = np.dot(X2, self.w.T)
        y1_new = [1 for i in range(len(X1))]
        y2_new = [0 for i in range(len(X2))]
        return x1_new,x2_new,y1_new,y2_new

    def predict(self, X):
        resultArray = []
        for i in range(len(X)):
            resultArray.append(np.dot(self.w, X[i]))
        print(resultArray)
        return np.array(resultArray)

    def score(self,X,Y):
        y_pred = self.predict(X)
        y_diff = Y - y_pred
        bingo = 0
        bad = 0
        for number in y_diff[0][0]:
            if int(number) == 0:
                bingo += 1
            else:
                bad += 1
        print("LDA分类的准确率为:{}".format(bingo/len(y_diff)))

if __name__ == "__main__":
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
    cls = LDA()
    X1_new, X2_new, y1_new, y2_new = cls.fit(trainx,trainy)
    cls.score(testx,testy)
    plt.figure()
    plt.plot(X1_new, y1_new, 'bo')
    plt.plot(X2_new, y2_new, 'ro')
    plt.show()

    # X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
    #                            n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    # cls = LDA()
    # X1_new, X2_new, y1_new, y2_new = cls.fit(X, y)
    # # 可视化原始数据
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    # plt.figure()
    # plt.plot(X1_new, y1_new, 'bo')
    # plt.plot(X2_new, y2_new, 'ro')
    # plt.show()




