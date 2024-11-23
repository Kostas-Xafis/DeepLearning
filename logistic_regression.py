import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import imageio
import os

class LogisticRegressionEP34:
    def __init__(self, lr=0.01):
        self.w = None    # Initialize weights to None; will set in `init_parameters()`
        self.b = None    # Initialize bias to None; will set in `init_parameters()`
        self.lr = lr     # Learning rate, default is 0.01
        self.f = None    # Placeholder for predictions in `forward`
        self.N = None    # Number of samples in the dataset
        self.gifImages = []

    def init_parameters(self, feature_dim):
        '''Initialize weights and bias with small random values'''
        self.w = random.randn(feature_dim) * 0.1
        self.b = 0

    def __lb(self, X):
        # Compute the linear combination of weights and features
        return X @ self.w + self.b

    def forward(self, X):
        self.f = np.power(1 + np.exp(-self.__lb(X)), -1)  # Sigmoid function
        return self.f

    def predict(self, X):
        return np.power(1 + np.exp(-self.__lb(X)), -1)

    def loss(self, X, y):
        # Compute predictions
        predictions = self.predict(X)
        # Binary cross-entropy loss
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def backward(self, X, y):
        # Gradient of loss w.r.t weights and bias
        error = self.f - y  # Error term, where `self.f` is computed by `forward()`
        self.l_grad_w = X.T @ error / len(y)  # Gradient for weights
        self.l_grad_b = np.mean(error)        # Gradient for bias

    def step(self):
        # Update weights and bias using learning rate
        self.w -= self.lr * self.l_grad_w
        self.b -= self.lr * self.l_grad_b

    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False, genGif=False):
        # Ensure input is numpy and initialize parameters
        X = np.array(X)
        y = np.array(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimension mismatch between X and y.")

        self.init_parameters(X.shape[1])

        # Shuffle and Batch data if batch_size is provided
        if batch_size is not None:
            shuffled_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffled_indices)
            X = X[shuffled_indices]
            y = y[shuffled_indices]

        self.N = X.shape[0]

        for i in range(1, iterations + 1):
            if batch_size is None:
                X_batch = X
                y_batch = y
            else:
                start = ((i - 1) * batch_size) % self.N
                end = start + batch_size
                if end > self.N:
                    # Avoid going out of bounds
                    end = self.N
                X_batch = X[start:end]
                y_batch = y[start:end]

            # Forward, backward, and step
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
            self.step()

            # Print loss periodically
            if i % show_step == 0:
                loss_value = self.loss(X, y)
                print(f"Iteration {i}, Loss: {loss_value}")

                if show_line:
                    self.show_line(X, y, i)
                    if genGif:
                        self.populateGIF(int(i / show_step))
        if genGif:
            self.generateGIF("logistic_regression.gif")


    def show_line(self, X, y, iterations):
        if (X.shape[1] != 2):
            print("Not plotting: Data is not 2-dimensional")
            return
        plt.ion()
        # plt.clf()

        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b-')
        plt.title(f"Iteration: {iterations}")
        plt.show(block=False)
        plt.pause(0.1)

    def populateGIF(self, i):
        plt.savefig(f"./images/temp_{i}.png")
        plt.clf()
        self.gifImages.append(imageio.imread(f"./images/temp_{i}.png"))

    def generateGIF(self, filename):
        imageCount = len(self.gifImages)
        # Add some extra frames to the end to make the gif, to extend it's time
        self.gifImages = self.gifImages + [self.gifImages[-1]] * 6
        imageio.mimsave(filename, self.gifImages, loop=0, fps=3)

        # Clean up temporary files
        for i in range(1, imageCount):
            os.remove(f"./images/temp_{i}.png")
