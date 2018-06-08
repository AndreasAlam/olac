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

    accuracy = np.equal(predictions[:, 0].round(), labels).mean()

    return predictions[:, 0].round(), accuracy

