import numpy as np

class LinearRegression:
    
    def __init__(self , lr = 0.001  , n_iters = 1000):
        
        self.lr  = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self , x_train , y_train):
        n_samples , n_features = x_train.shape
        
        
        #Initial weights and biases
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #gradient descend
        for _ in range(self.n_iters):
            
            linear_model = np.dot(self.weights , x_train.T) + self.bias
            
            dw = (1 / n_samples) * (np.dot((linear_model - y_train) , x_train))
            db = (1/ n_samples) * np.sum((linear_model - y_train))
            
            self.weights -=  self.lr * dw
            self.bias -= self.lr * db

        
        
    
    def predict(self , x_test):
         linear_model = np.dot(self.weights , x_test.T) + self.bias
         return linear_model
    
    
    def accuracy(self , predicted , y_test):
        return np.mean((y_test - predicted) ** 2)
    
        