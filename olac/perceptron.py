import math
import numpy as np
# [RU] Why use the built-in random lib when you have np already loaded
from random import random


def PointsInCircum(r, n=100):
    return np.array([(math.cos(2*np.pi/n*x)*r, math.sin(2*np.pi/n*x)*r) for x in range(0, n+1)])


def PointsInCircum2(r, n=100):
    points = []
    for x in range(n):
        points.append((math.cos(2*np.pi/n*x)*(r+random()*(r/2)), math.sin(2*np.pi/n*x)*(r+random()*(r/2))))
    return np.array(points)


def rand_cluster(n, c, r):
    """returns n random points in disk of radius r centered at c"""
    x, y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random()
        s = r*random()
        points.append((x+s*math.cos(theta), 6*y+s*math.sin(theta)))
    return np.array(points)


def generate_data(n_samples):
    # n_samples = 1000
    X1 = PointsInCircum2(40, int(n_samples/2))
    X2 = rand_cluster(int(n_samples/2), (0, 0), 30)
#     plt.scatter(*X1.T)
#     plt.scatter(*X2.T)

    X = np.vstack([X1, X2])
    y0 = np.zeros(shape=(int(n_samples/2), 1))
    y1 = np.ones(shape=(int(n_samples/2), 1))
    yhat = np.vstack([y0, y1])
    return X, yhat


class Perceptron(object):

    def __init__(self):
        self.weights = []

    def sig(self, z):
        return 1 / (1 + np.exp(-z))

    def dsig_dz(self, z):
        return self.sig(z) * (1 - self.sig(z))

    def J(self, y, yhat):
        eps = 1e-8
        return -(yhat*np.log(y+eps) + (1-yhat)*np.log(1-y+eps))

    def dJ_dy(self, y, yhat):
        eps = 1e-8
        return (1-yhat)/(1-y+eps) - yhat/(y+eps)

    def relu(self, z):
        return np.where(z > 0, z, 0)

    def drelu_dz(self, z):
        return np.where(z > 0, 1, 0)

    def backwardJ(self, x0, w1, b1, w2, b2, y, yhat, alpha):
        # quantities
        z1 = np.dot(x0, w1) + b1.T
        x1 = self.relu(z1)
        z2 = np.dot(x1, w2) + b2.T
        # y = sig(z2)

        delta2 = self.dJ_dy(y, yhat) * self.dsig_dz(z2)
        delta1 = np.matmul(w2, delta2) * self.drelu_dz(z1).T

        w2 -= alpha * np.multiply(delta2, x1).T
        w1 -= alpha * np.multiply(delta1, x0).T

        b2 -= alpha * delta2
        b1 -= alpha * delta1

        return w1, b1, w2, b2

    def predict(self, x0, w1, b1, w2, b2):
        x1 = self.relu(np.dot(x0, w1) + b1.T)  # output of hidden layer
        return self.sig(np.dot(x1, w2) + b2.T)  # output of output layer

    def train_network(self, X, labels, step, nr_epochs, n_hidden):
        # implementation of pseudocode
        # initialize weights
        # n_samples = X.shape[0]
        n_input = X.shape[1]
        n_out = 1
        costs = np.zeros(shape=(nr_epochs, 1))
        y_hat_save = np.zeros(shape=(nr_epochs, len(labels)))
    #    X = input data
        w1 = np.random.normal(0, 0.1, size=(n_input, n_hidden))
        w2 = np.random.normal(0, 0.1, size=(n_hidden, n_out))

        b1 = np.random.normal(0, 0.1, size=(n_hidden, 1))
        b2 = np.random.normal(0, 0.1, size=(n_out, 1))

        for epoch in range(nr_epochs):
            for i, row in enumerate(X):
                y_pred = self.predict(row, w1, b1, w2, b2)
                w1, b1, w2, b2 = self.backwardJ(row, w1, b1, w2, b2, y_pred, labels[i], step)

            curr_pred = self.predict(X, w1, b1, w2, b2)
            costs[epoch] = np.mean(self.J(curr_pred, labels))
            y_hat_save[epoch] = np.squeeze(curr_pred.round())

            if ((epoch % 10) == 0) or (epoch == (nr_epochs - 1)):
                # print(curr_pred.round())
                # print((labels == curr_pred.round()).sum())
                accuracy = np.mean(np.equal(curr_pred[:, 0].round(), labels[:, 0]))  # current accuracy on entire set
                print('Training accuracy after epoch {}: {:.4%}'.format(epoch, accuracy))

        return w1, b1, w2, b2, costs, y_hat_save

    def fit_model(self, X, labels, epochs, step, n_hidden):
        # implementation of pseudocode
        # initialize weights
        # n_samples = X.shape[0]
        n_input = X.shape[1]
        n_out = 1

    #    X = input data
        w1 = np.random.normal(0, 0.1, size=(n_input, n_hidden))
        w2 = np.random.normal(0, 0.1, size=(n_hidden, n_out))

        b1 = np.random.normal(0, 0.1, size=(n_hidden, 1))
        b2 = np.random.normal(0, 0.1, size=(n_out, 1))

        for epoch in range(epochs):
            for i, row in enumerate(X):
                y_pred = self.predict(row, w1, b1, w2, b2)
                w1, b1, w2, b2 = self.backwardJ(row, w1, b1, w2, b2, y_pred, labels[i], step)

            # curr_pred = self.predict(X, w1, b1, w2, b2)

            # if ((epoch % 10) == 0) or (epoch == (epochs - 1)):
            #     # print(curr_pred.round())
            #     # print((labels == curr_pred.round()).sum())
            #     accuracy = np.mean(np.equal(curr_pred[:, 0].round(), labels[:, 0]))  # current accuracy on entire set
            #     print('Training accuracy after epoch {}: {:.4%}'.format(epoch, accuracy))

        return w1, b1, w2, b2
