from sklearn import tree
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)
class MarvellousKNN():
    def fit(self,TrainingData,TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget
    def predict(self,TestData):
        prediction = []
        for row in TestData:
            lebel = self.closest(row)
            prediction.append(lebel)
        return prediction
    def closest(self,row):
        bestdistance = euc(row,self.TrainingData[0])
        bestindex = 0
        for i in range (1,len(self.TrainingData)):
            dist = euc(row,self.TrainingData[i])
            if dist < bestdistance:
                bestindex = i
        print("the indesx is %d"% self.TrainingTarget[bestindex] ,"distance %d"%dist)
        return self.TrainingTarget[bestindex]
def MarvellousKneighbour():
    border = "-"*50
    iris = load_iris()
    data = iris.data
    target = iris.target
    print(border)
    print("actual data set")
    print(border)
    for i in range(len(iris.target)):
        print("ID: %d,Lable %s,Feature:%s"%(i,iris.data[i],iris.target[i]))
    print("size of actual data set %d"%(i+1))
    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.3
                                                                     )
    print(border)
    print("training data set ")
    print(border)
    for i in range(len(data_test)):
        print("ID: %d ,lable %s ,feature: %s"%(i,data_train[i],target_train[i]))
    print('size of training dataset %d'%(i+1))
    print(border)
    classifier = MarvellousKNN()
    classifier.fit(data_train,target_train)
    prediction = classifier.predict(data_test)
    Accuracy = accuracy_score(target_test,prediction)
    return Accuracy
def main():
    Accuracy =MarvellousKneighbour()
    print("Accuracy of classification algorithm with k neighbor",Accuracy*100,"%,")
if __name__ =="__main__":
    main()