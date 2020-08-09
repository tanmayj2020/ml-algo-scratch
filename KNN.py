import numpy as np


from collections import Counter
#using only the euclidean distance and not considering other distances for now
#this function calculates the distance between two points
def euclidean_distance(x1 , x2):
    return np.sqrt(np.sum((x2 - x1)**2))
   



class KNN:
    
    def __init__(self , k = 3):
        self.k = k
    
    def fit( self, X , y):
        self.X = X 
        self.y = y 
    
    
    def predict(self , X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self , x):
        
        # calculate the distance of all datasamples in training set
        
        distances = [euclidean_distance(x , x_train) for x_train in self.X ]
        indexes_sorted = np.argsort(distances)
        
        #calculate the labels of k nearestneighbour
        k_nearest_points_label = [self.y[i] for i in indexes_sorted][:self.k]
        
        #return the most common label 
        most_common_label = Counter(k_nearest_points_label).most_common(1)
        #as counter returns list of tuples        
        return most_common_label[0][0]
    
    def accuracy(self , predictions , y_test):
        acc = np.sum(predictions == y_test) / len(y_test)
        return acc
    
