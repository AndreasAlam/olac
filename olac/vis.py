import time
import numpy as np
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt

from . import maths as mf
from . import utils as ut
import imageio
# [RU] imported but unused
# from olac.perceptron import Perceptron as pc


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

        labels = labels.reshape(predictions.shape)
        assert predictions.shape == labels.shape

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
        except TypeError:
            w1, b1, w2, b2 = weights
            predictions = model.predict(data, w1, b1, w2, b2)

        labels = labels.reshape(predictions.shape)
        TP = np.equal(predictions[predictions.round() == 1].round(), labels[predictions.round() == 1]).mean()
        # TN = np.equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        FP = np.not_equal(predictions[predictions.round() == 1].round(), labels[predictions.round() == 1]).mean()
        # FN = np.not_equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        return predictions.round(), TP/(TP+FP)

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
        except TypeError:
            w1, b1, w2, b2 = weights
            predictions = model.predict(data, w1, b1, w2, b2)

        labels = labels.reshape(predictions.shape)
        TP = np.equal(predictions[predictions.round() == 1].round(), labels[predictions.round() == 1]).mean()
        # TN = np.equal(predictions[predictions.round() == 0, 0].round(), labels[:, 0]).mean()
        # FP = np.not_equal(predictions[predictions.round() == 1, 0].round(), labels[:, 0]).mean()
        FN = np.not_equal(predictions[predictions.round() == 0].round(), labels[predictions.round() == 0]).mean()
        return predictions.round(), TP/(TP+FN)


def get_fun_map(xlim, ylim, weights, MODEL):

    try:
        w1, b1, w2, b2 = weights
        x1 = np.linspace(xlim[0], xlim[1], 100)
        x2 = np.linspace(ylim[0], ylim[1], 100)
        fun_map = np.empty((x1.size, x2.size))

        for nn, ii in enumerate(x1):
            for mm, jj in enumerate(x2):
                fun_map[mm, nn] = MODEL.predict([ii, jj], w1, b1, w2, b2)
    
    except TypeError:
        x1 = np.linspace(xlim[0], xlim[1], 100)
        x2 = np.linspace(ylim[0], ylim[1], 100)
        fun_map = np.meshgrid(x1, x2)
        fun_map = MODEL.predict(np.reshape(fun_map, (2, 100*100)).T)
        fun_map = fun_map.reshape(100, 100)

    return fun_map


def main(MODEL, GENERATOR, metric, weights=None, window=20, p_train=10, write=False, **kwargs):
    """
    Display the model accuracy, recall or precision over time together with the data points as predicted
    by the model.

    Parameters
    ----------

        MODEL : Object
            Keras model or Perceptron Object as made by us

        GENERATOR : generator object?
            Generator that simulates the datapoints. For example popping clusters or roving balls

        metric : string
            "accuracy", "precision" or "recall": Which metric you want to display over time.

        weights : ndarray
            If you have a pretrained perceptron, you can input the learned weights here. For a keras
            model this should be saved in the model object.

        window : int
            Size of the sliding window: How many datapoints you want to predict at the same time

        p_train : int
            percentage of datapoints you want to use for training

        **kwargs
            Input for the generator
    """
    var_c = "N"
    while var_c == "N":
        # Initialize
        data = np.array(list(GENERATOR(**kwargs)))
        itrs = len(data)
        pnts = data[:, :2]
        pnts = ut.data_prep(pnts)
        lbls = data[:, -1]
        wndw = int(itrs/p_train)

        xlim = (min(pnts[:, 0]) + min(pnts[:, 0])/10, max(pnts[:, 0])+max(pnts[:, 0])/10)
        ylim = (min(pnts[:, 1]) + min(pnts[:, 1])/10, max(pnts[:, 1])+max(pnts[:, 1])/10)

        if weights:
            # dont train model
            pass
        else:
            # train model
            try:
                weights = MODEL.fit(pnts[:wndw], lbls[:wndw], epochs=200, batch_size=32, verbose=0)

            except AttributeError:
                print("Training model")
                weights = MODEL.fit_model(pnts[:wndw], lbls[:wndw], epochs=500, step=0.001, n_hidden=3)

        # Plot the first trained batch to see if it is doing OK
        fun_map = get_fun_map(xlim, ylim, weights, MODEL)
        plt.figure(figsize=(20, 10))
        plt.imshow(fun_map, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   vmin=0, vmax=1, aspect='auto', origin='lower')
        new_pred, metric_value = getattr(GetNewMetric, 'get_new_' + metric)(
                                         pnts[:wndw, :],
                                         lbls[:wndw],
                                         MODEL,
                                         weights)
        plt.scatter(*pnts[:wndw, :][np.not_equal(new_pred[:, 0], lbls[:wndw]), :].T, color='r')
        plt.scatter(*pnts[:wndw, :].T, color=[{0: 'w', 1: 'k', 2: 'b'}[n] for n in lbls[:window].astype(int)],
                    alpha=.6)
        plt.show()

        print("Contiue? Y/N")
        var_c = input().upper()
        if var_c == "N":
            weights = None

    metric_in_time = []
    for i in np.arange(1, itrs, int(window/4)):

        pnts_window = pnts[i:window+i]
        label_window = lbls[i:window+i]

        new_pred, metric_value = getattr(GetNewMetric, 'get_new_' + metric)(pnts_window,
                                                                            label_window,
                                                                            MODEL,
                                                                            weights)
        assert new_pred[:, 0].shape == label_window.shape

        metric_in_time.append(metric_value)

        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.plot(np.arange(0, len(metric_in_time)), metric_in_time)
        plt.xlim(0, int(len(np.arange(1, itrs, int(window/4)))))
        plt.ylim(0, 1.1)
        plt.title(metric+'{:.4%}'.format(metric_value))

        plt.subplot(122)
        fun_map = get_fun_map(xlim, ylim, weights, MODEL)
        plt.imshow(fun_map, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   vmin=0, vmax=1, aspect='auto', origin='lower')
        plt.scatter(*pnts_window.T, color=[{0: 'w', 1: 'k', 2: 'b'}[n] for n in label_window.astype(int)],
                    alpha=.8)
        plt.scatter(*pnts_window[np.not_equal(new_pred[:, 0], label_window), :].T, color='r')
        plt.xlim(xlim)
        plt.ylim(ylim)

        display.clear_output(wait=True)
        display.display(plt.gcf())

        if write:
            plt.savefig('tmp/'+str(i)+'.png')
        plt.close()

    # if write:
    #     images = []
    #     for i in np.arange(1, itrs, int(window/4)):
    #         filename = 'tmp/'+str(i)+'.png'
    #         images.append(imageio.imread(filename))
    #         imageio.mimsave('progressgif'+time.strftime('%Y-%m-%d', time.localtime(time.time()))+'.gif', images)

########################################################################################################################
#                                                    Learning at Cost                                                  #
########################################################################################################################


def plot_linear_ls(x, y, window_size=10, constant=True, colour='r', label=None):
    """Plot the linear fits for each period

    Parameters
    ----------
    x : ndarray
        The independent variables, features, over which to fit
    y : ndarray
        The dependent variable which we want approximate
    window_size : int (optional)
        The number of observations that will included in a period
    constant : boolean (optional)
        Whether a constant should be added, default is True
    colour : str
        The colour paramter to be passed to pyplot
    label : str
        The label parameter to be passed to pyplot

    Returns
    -------
    coefs : ndarray
        An numpy array containing the coefficients.
    ind : list
        The indexes of the periods
    """
    coefs, ind = mf.seq_linear_ls(x=x, y=y, window_size=window_size, constant=constant)
    for i in range(len(ind)):
        xi = ind[i]
        alpha, beta = coefs[i]
        plt.plot(xi, alpha + beta * xi, c=colour, label=label)
    return coefs, ind
