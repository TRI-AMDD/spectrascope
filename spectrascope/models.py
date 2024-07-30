import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, l1_penalty=0.1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 1. Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 2. Perform gradient descent
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2/num_samples) * np.dot(X.T, (y_predicted - y)) + self.l1_penalty * np.sign(self.weights)
            db = (2/num_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)