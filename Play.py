import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

def play_predictor(data_path):
    data = pd.read_csv(data_path)
    features =["Wether,Temperature"]
    print("The feature are ",features)
    whether = data.Wether
    temp = data.Temperature
    play = data.Play
    le = preprocessing.LabelEncoder()
    temp_encoding = le.fit_transform(temp)
    wheat_encoding = le.fit_transform(whether)
    label = le.fit_transform(play)
    Feature = list(zip(wheat_encoding,temp_encoding))
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(Feature,label)
    prediction = model.predict([[2,0]])
    print(prediction)
def main():
    print("___Vaibhav infosystem by vm____")
    print("Machine learning Application")
    play_predictor("C:\\Users\HP\Desktop\python assignments\MarvellousInfosystems_PlayPredictor.csv")
if __name__=="__main__":
    main()
