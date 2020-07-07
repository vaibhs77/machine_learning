import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

def Wine_predictor(data_path):
    data = pd.read_csv(data_path)
    Feature = ["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280 / OD315 of diluted wines","Proline"]
    print("The Feature name are",Feature)
    a = data.Alcohol
    b= data.Malic_acid
    c = data.Ash
    d = data.Alcalinity_of_ash
    e =data.Magnesium
    f= data.Total_phenols
    g= data.Flavanoids
    h = data.Nonflavanoid_phenols
    x = data.Proanthocyanins
    y =data.Color_intensity
    z = data.Hue
    s =data.Proline

    c1 = data.Class
    label=list(zip(c1))
    features = list(zip(a,b,c,d,e,f,g,h,x,y,z,s))
    print(features,label)
    x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.9)
    model = KNeighborsClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(y_pred)
    print(y_test)
    print("Accuracy",metrics.accuracy_score(y_test,y_pred))



def main():
    print("The wine predictor ")
    Wine_predictor ( "C:\\Users\HP\Desktop\python assignments\WinePredictor - WinePredictor (2).csv" )
main()