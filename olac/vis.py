import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns
import numpy as np
import time

from olac.perceptron import Perceptron as pc


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
        except TypeError:
            w1, b1, w2, b2 = weights
            predictions = model.predict(data, w1, b1, w2, b2)
        
        accuracy = np.equal(predictions.round(), labels).mean()

        return predictions.round(), accuracy

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


def main_function(MODEL, GENERATOR, iterations, metric, weights=None, window=20, **kwargs):
    """

    """
    data = np.array(list(GENERATOR(steps=iterations, **kwargs)))
    X = data[:, :2]
    labels = data[:, -1]
    wdw = int(iterations*0.1)

    xlim = (min(X[:, 0]) + min(X[:, 0])/40, max(X[:, 0])+max(X[:, 0])/40)
    ylim = (min(X[:, 1]) + min(X[:, 1])/40, max(X[:, 1])+max(X[:, 1])/40)
    if weights:
        # dont train model
        pass
    else:
        # train model
        try:
            weights = MODEL.fit(X[:wdw], labels[:wdw], epochs=200, batch_size=32)
        except AttributeError:
            # model = MODEL
            weights = MODEL.fit_model(X[:wdw], labels[:wdw], epochs=500, step=0.001, n_hidden=3)

        print("Training model")
        # get weights

    metric_in_time = []
    for i in np.arange(1, iterations, int(window/4)):

        X_window = X[i:window+i]
        label_window = labels[i:window+i]

        new_pred, metric_value = getattr(GetNewMetric, 'get_new_' + metric)(X_window, label_window, MODEL, weights)
        metric_in_time.append(metric_value)

        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.plot(np.arange(0, len(metric_in_time)), metric_in_time)
        plt.xlim(0, int(len(np.arange(1, iterations, int(window/2)))))
        plt.ylim(0, 1.1)
        plt.title(metric+'{:.4%}'.format(metric_value))

        plt.subplot(122)
        x1 = np.linspace(xlim[0], xlim[1], 100)
        x2 = np.linspace(ylim[0], ylim[1], 100)
        fun_map = np.empty((x1.size, x2.size))
        for nn, ii in enumerate(x1):
            for mm, jj in enumerate(x2):
                w1, b1, w2, b2 = weights
                fun_map[mm, nn] = MODEL.predict([ii, jj], w1, b1, w2, b2)
        plt.imshow(fun_map, extent=[x1.min(), x1.max(), x2.min(), x2.max()],
               vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.scatter(*X_window.T, color=[{0: 'w', 1: 'k', 2: 'b'}[n] for n in label_window.astype(int)], alpha=.3)
        # plt.scatter(*X[new_pred == 0, :].T, color='w', alpha=.3)
        plt.scatter(*X_window[np.not_equal(new_pred[:, 0], label_window), :].T, color='r')
        plt.xlim(xlim)
        plt.ylim(ylim)
        # x1 = np.linspace(0, 1000, 100)
        # x2 = np.linspace(0, 1000, 100)

        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.close()
