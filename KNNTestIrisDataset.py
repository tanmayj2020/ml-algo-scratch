from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X , y = iris.data , iris.target
 
x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size = 0.2 , random_state = 1234)
 
 
from KNN import KNN
 
clf = KNN(k=5)
 
clf.fit(x_train , y_train)
predicted = clf.predict(x_test)
 
accuracy = clf.accuracy(predicted , y_test)