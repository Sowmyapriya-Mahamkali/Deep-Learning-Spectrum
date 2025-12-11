import numpy as np

# -------------------
# Perceptron
# -------------------
class Perceptron:
    def __init__(self, input_dim, lr=0.01, epochs=10):
        self.W = np.zeros((input_dim,1))
        self.b = 0
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = np.dot(xi, self.W) + self.b
                update = self.lr * (yi - np.sign(pred))
                self.W += update * xi.reshape(-1,1)
                self.b += update

    def predict(self, X):
        return np.sign(X @ self.W + self.b)

# -------------------
# Linear SVM (Hinge Loss)
# -------------------
class LinearSVM:
    def __init__(self, input_dim, lr=0.001, C=1.0, epochs=10):
        self.W = np.zeros((input_dim,1))
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def hinge_loss(self, X, y):
        distances = 1 - y * (X @ self.W + self.b)
        distances[distances < 0] = 0
        return 0.5 * np.sum(self.W**2) + self.C * np.sum(distances)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                if yi * (xi @ self.W + self.b) < 1:
                    self.W += self.lr * (xi.reshape(-1,1) * yi - self.W)
                    self.b += self.lr * yi
                else:
                    self.W -= self.lr * self.W

    def predict(self, X):
        return np.sign(X @ self.W + self.b)
