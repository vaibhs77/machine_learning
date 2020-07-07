import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.preprocessing  import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def sale_prediction(datasheet):
    data = pd.read_csv(datasheet)
    print(data)
    TV = data["TV"].values
    radio = data["radio"].values
    newspaper = data["newspaper"].values
    sale= data["sales"].values
    X =list(zip(TV,radio,newspaper))
    print(X)
    Y = list(zip(sale))
    print(Y)
    clf = LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split( X, Y, test_size=0.3, random_state=0)
    x= clf.fit(x_train,y_train)
    y_pred =clf.predict(x_test)
    print("the value after prediction is ",y_pred , x_test)
    Accuracy = accuracy_score ( x_test,  y_pred.round(), normalize=False)
    return Accuracy


def main ():
    datasheet = "C:\\Users\HP\Desktop\python assignments\Advertising.csv"
    Accuracy = sale_prediction (datasheet)
    print ( "Accuracy of classification algorithm with k neighbor", Accuracy * 100, "%," )


if __name__ == "__main__":
    main ()



