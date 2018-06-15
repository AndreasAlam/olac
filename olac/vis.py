import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns
import numpy as np
import time

import olac.perceptron as pc


def demo_plot():
    # Initialize
    start = [100]
    N = [0]

    rs = np.random.RandomState(9)
    x, y = rs.multivariate_normal([0, 0], [(1, 0), (0, 2)], 100).T
    x1, y1 = rs.multivariate_normal([80, 70], [(1, 0), (0, 2)], 100).T

    for n in range(20):

        plt.subplot(121)
        sns.despine()

        # Calculate next step of performance.
        # Here is where the actual output of the model will be placed
        start.append(start[n]-np.random.random(1)[0]*10)
        N.append(n)

        # plot performance
        plt.plot(N, start, 'r-')
        plt.xlim((0, 100))
        plt.ylim((0, 100))
        plt.title('Performance '+str(np.round(start[n], 2)))

        plt.subplot(122)
        sns.despine()

        # Calculate next step of clusters
        x += np.random.random(size=x.shape)*2
        y += np.random.random(size=y.shape)*2

        x1 -= np.random.random(size=x1.shape)*2
        y1 -= np.random.random(size=y1.shape)*2

        # plot the clusters
        plt.plot(x, y, '.')
        plt.plot(x1, y1, '.')
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        # Update the display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(.1)

        # Add close, so it runs faster and the final plot isn't made cuz we don't need that in our life
        plt.close()

# todo: get_new_recall, precicion etc


class GetNewMetric:

    @staticmethod
    def get_new_accuracy(data, labels, model=None, weights=None):
        """
        Get the accuracy and stuff back to display how it changes over time

        Parameters
        ----------

        data : ndarray | shape (N, 2)
            New clusters to classify

        model : Model class
            Keras based deep learning model class

        labels : ndarray | shape (N, )
            Labels of the datapoints.

        weights : tuple
            w1, b1, w2, b2

        """

        try:
            predictions = model.predict(data)
        except AttributeError:
            w1, b1, w2, b2 = weights
            predictions = pc.prediction(data, w1, b1, w2, b2)

        accuracy = np.equal(predictions[:, 0].round(), labels[:, 0]).mean()

        return predictions[:, 0].round(), accuracy

    @staticmethod
    def get_new_precision(data, labels, model=None, weights=None):
        """
        Get the accuracy and stuff back to display how it changes over time
        Assumes 1 is the positive label!!
        Parameters
        ----------

        data : ndarray | shape (N, 2)
            New clusters to classify

        model : Model class
            Keras based deep learning model class

        labels : ndarray | shape (N, )
            Labels of the datapoints.

        weights : tuple
            w1, b1, w2, b2

        """

        try:
            predictions = model.predict(data)
        except AttributeError:
            w1, b1, w2, b2 = weights
            predictions = pc.prediction(data, w1, b1, w2, b2)

        TP = np.equal(predictions[predictions.round() == 1, 0].round(), labels[:, 0]).mean()
        # TN = np.equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        FP = np.not_equal(predictions[predictions.round() == 1, 0].round(), labels[:, 0]).mean()
        # FN = np.not_equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        return predictions[:, 0].round(), TP/(TP+FP)

    @staticmethod
    def get_new_recall(data, labels, model=None, weights=None):
        """
        Get the accuracy and stuff back to display how it changes over time
        Assumes 1 is the positive label!!
        Parameters
        ----------

        data : ndarray | shape (N, 2)
            New clusters to classify

        model : Model class
            Keras based deep learning model class

        labels : ndarray | shape (N, )
            Labels of the datapoints.

        weights : tuple
            w1, b1, w2, b2

        """

        try:
            predictions = model.predict(data)
        except AttributeError:
            w1, b1, w2, b2 = weights
            predictions = pc.prediction(data, w1, b1, w2, b2)

        TP = np.equal(predictions[predictions.round() == 1, 0].round(), labels[:, 0]).mean()
        # TN = np.equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        # FP = np.not_equal(predictions[predictions.round() == 1, 0].round(), labels[:, 0]).mean()
        FN = np.not_equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        return predictions[:, 0].round(), TP/(TP+FN)


def main_function(data, MODEL, GENERATOR, iterations, metric, weights=None, **kwargs):
    """

    """
    if weights:
        # dont train model\

    else:
        # train model
        GETFIRST_DATA_TO_TRAIN_MODEL
        print("Training model")
        # get weights

    for i, x in enumerate(generator()):

        data_point = whatever
        data = whatever

        metric_in_time = []
        new_pred, metric = getattr(GetNewMetric, 'get_new_' + metric)
        metric_in_time.append(metric)


