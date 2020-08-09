from sklearn import datasets
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)





from LinearRegression import LinearRegression

regressor = LinearRegression(lr = 0.1)
regressor.fit(x_train , y_train)



predict_test = regressor.predict(x_test)
error_test = regressor.accuracy(predict_test , y_test)

predict_train = regressor.predict(x_train)
error_train = regressor.accuracy(predict_train , y_train)


print("The test error of model is {} and train error of model is {}".format(accuracy_test , accuracy_train))


#Inspect data
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30) 
plt.show()


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()
