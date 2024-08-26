import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # get Y_pred first
            y_pred = np.dot(X, self.weights) + self.bias
            
            # get the update in weights and biases
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update the weights and biases
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    
    def fit_two(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # get Y_pred first
            y_pred = np.dot(X, self.weights) + self.bias
            
            # dw = []
            # db = []
            for i in range(n_samples):

                # get the update in weights and biases
                dw_i = (1/n_samples) * (X[i] * (y_pred[i] - y[i]))
                
                # update the weights and biases
                for j in range(n_features):
                    self.weights[j] = self.weights[j] - self.lr * dw_i
            
            db = (1/n_samples) * np.sum(y_pred - y)
            self.bias = self.bias - self.lr * db

    def predict(self, X):
         predictions = np.dot(X, self.weights) + self.bias
         return predictions

