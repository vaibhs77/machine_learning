import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
print("feature name of iris data set")
print(iris.feature_names)
print("target name of iris data set")
print(iris.target_names)

# indices removing from list
test_index = [1,51,101]
#traing data with removed element
train_target = np.delete(iris.target,test_index)
train_data =np.delete(iris.data,test_index,axis=0)


#Testing data for testing on training data
test_target =iris.target[test_index]
test_data=iris.data[test_index]

#dsion tree classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data,train_target)
print("the value we removed for testing is %s"%(test_target))
print("the result of predicti is%s"%classifier.predict(test_data))
#visualisation
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(classifier,out_file=dot_data,feature_names=iris.feature_names,class_names =iris.target_names,filled=True,impurity= False)
graph= pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write("marvellous.pdf")