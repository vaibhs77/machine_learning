import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Headbraincheck():
    data = pd.read_csv("MarvellousHeadBrain.csv")
    print("size of data set ", data.shape)
    X = data['Head _Size(cm^3)'].values
    print(data)
    Y = data["Brain_Weight(grams)"].values

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)
    numerator = 0
    denomenator = 0
    for i in range(n):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denomenator += (X[i] - mean_x) ** 2
        m = numerator / denomenator
        c = mean_y - (m * mean_x)
        max_x = np.max(X) + 100
        min_x = np.min(X) - 100
        x = np.linspace(min_x, max_x, n)
        y = c + m * x
        plt.plot(x, y, color="#58b970", label="Regression list")
        plt.scatter(X, Y, color="#ef5423", label="scatter plot")
        plt.xlabel("head size in cm3")
        plt.ylabel("brain weight in gram")
        plt.legend()
        plt.show()

        ss_t = 0
        ss_r = 0
        for i in range(n):
            y_pred = c + m * X[i]
            ss_t += (Y[i] - mean_y) ** 2
            ss_r += (Y[i] - y_pred) ** 2
            r2 = 1 - (ss_r / ss_t)
            print(r2)
def main():
    Headbraincheck()
if __name__ == "__main__":
    main()
