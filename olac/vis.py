import time
import numpy as np
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imageio
import pandas as pd

from . import maths as mf
from . import utils as ut

from sklearn.base import clone
import os

# [RU] imported but unused
# import imageio

# [RU] imported but unused
# from olac.perceptron import Perceptron as pc


def performance(eval_data, train_data, window=10):
    plt.figure(figsize=(20, 20))
    sns.set_style('white')
    sns.set_context('talk', font_scale=2)
    clrs = sns.palettes.mpl_palette(name='Set2', n_colors=4)

    df_eval = ut.queue_point_list_to_df(eval_data)
    df_train = ut.queue_point_list_to_df(train_data)

    df_eval.fillna(99)
    df_eval.fillna(99)

#     for n, thing in ['Accuracy', 'Precision', 'Recall']:
    df_train['TP'] = (df_train['y_pred'] == 1) & (df_train['y_true'] == 1)
    df_train['FP'] = (df_train['y_pred'] == 1) & (df_train['y_true'] == 0)
    df_train['TN'] = (df_train['y_pred'] == 0) & (df_train['y_true'] == 0)
    df_train['FN'] = (df_train['y_pred'] == 0) & (df_train['y_true'] == 1)

    df_eval['TP'] = (df_eval['y_pred'] == 1) & (df_eval['y_true'] == 1)
    df_eval['FP'] = (df_eval['y_pred'] == 1) & (df_eval['y_true'] == 0)
    df_eval['TN'] = (df_eval['y_pred'] == 0) & (df_eval['y_true'] == 0)
    df_eval['FN'] = (df_eval['y_pred'] == 0) & (df_eval['y_true'] == 1)

    r_accuracy_obs = (df_train['TP'].rolling(window).sum() +
                      df_train['TN'].rolling(window).sum())/window
    r_accuracy_act = (df_eval['TP'].rolling(window).sum() +
                      df_eval['TN'].rolling(window).sum())/window

    r_precision_act = (df_eval['TP'].rolling(window).sum() /
                       (df_eval['TP'].rolling(window).sum() +
                       df_eval['FP'].rolling(window).sum()))
    r_precision_obs = (df_train['TP'].rolling(window).sum() /
                       (df_train['TP'].rolling(window).sum() +
                        df_train['FP'].rolling(window).sum()))

    r_recall_act = (df_eval['TP'].rolling(window).sum() /
                    (df_eval['TP'].rolling(window).sum() +
                     df_eval['FN'].rolling(window).sum()))
    r_recall_obs = (df_train['TP'].rolling(window).sum() /
                    (df_train['TP'].rolling(window).sum() +
                     df_train['FN'].rolling(window).sum()))

    r_sens_act = (df_eval['TP'].rolling(window).sum()/window)
    r_sens_obs = (df_train['TP'].rolling(window).sum()/window)

    r_spec_act = (df_eval['TN'].rolling(window).sum()/window)
    r_spec_obs = (df_train['TN'].rolling(window).sum()/window)

    plt.subplot(221)
    plt.plot(r_accuracy_obs, c=clrs[0], label='Observed')
    plt.plot(r_accuracy_act, c=clrs[1], label='Unobserved')
    plt.legend()
    plt.title('Accuracy')
    plt.ylim(0, 1.05)

    plt.subplot(222)
    plt.plot(r_precision_obs, c=clrs[0], label='Observed')
    plt.plot(r_precision_act, c=clrs[1], label='Unobserved')
    plt.legend()
    plt.title('Precision')
    plt.ylim(0, 1.05)

    plt.subplot(223)
    plt.plot(r_recall_obs, c=clrs[0], label='Observed')
    plt.plot(r_recall_act, c=clrs[1], label='Unobserved')
    plt.legend()
    plt.title('Recall')
    plt.ylim(0, 1.05)

    plt.subplot(224)
    plt.plot(r_sens_act, c=clrs[0], label='Sensitivity Unobserved (TP)')
    plt.plot(r_spec_act, c=clrs[1], label='Specificity Unobserved (TN)')
    plt.plot(r_sens_obs, c=clrs[2], label='Sensitivity Observed (TP)')
    plt.plot(r_spec_obs, c=clrs[3], label='Specificity Observed (TN)')
    plt.legend()
    plt.title('TPR and TNR')
    plt.ylim(0, 1.05)

    plt.tight_layout()


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


class Plotter():
    """
    Plotter class to save convenient plot functions
    """
    def __init__(self, window=100, figsize=(10,5)):
        self.X = []
        self.y = []
        self.X_val = []
        self.cost = [0]
        self.acc = []
        self.acc2 = []
        self.cols = np.array(['g', 'b'])
        self.window = window
        self.filenames = []
        self.figsize = figsize


    def get_grid(self):
        """Return a grid that can be used
        to plot the decision function"""
        gridpoints = np.linspace(-10, 10, 250)
        grid = []
        for y in gridpoints:
            for x in gridpoints:
                grid.append([x, y])
        return np.array(grid)

    def plot_history(self, pipeline, test_set, train_set, save_gif=False, gen_name=None, trans=None,
                     plot_cost=False, budget=0, gain_factor=2, plot_step=1, sup_title=''):
        """Plot the training of the model and plot the accuracy and (optionally)
        the cost/profit over time.

        Parameters
        ----------

        pipeline : Pipeline
            Pipeline object used to run the scenario

        test_set: list
            output of the pipeline.run()

        save_gif: boolean
            Save as gif or not

        gen_name: string
            Name of the data_generator used. This will appear in the title and
            the saved gif name

        trans: Transformer
            transformer used to transform the input for example sklearn.kernel_approximation.RBFSampler

        plot_cost: boolean
            Whether or not to plot the cost

        budget: float/int
            Starting budget for the cost plot. Default == 0

        gain_factor: float/int
            How much we earn from catching fraud compared to investigation cost. The total gain
            is calculated as investigation_cost*gain_factor. The investigation cost is set to 1.

        plot_step: int
            Every plot_step steps an image is saved for the gif. Long runs can be shortend by using a
            higher plot_step

        sup_title: string
            Title of the entire plot
        """

        plt.rcParams['figure.figsize'] = self.figsize

        test_df = ut.queue_point_list_to_df(test_set)
        train_df = ut.queue_point_list_to_df(train_set)

        y_true = test_df['y_true']
        y_pred = test_df['y_pred']

        acc2 = np.array_split(y_true == y_pred, len(pipeline.predictor.model_hist))
        X_in = np.array_split(test_df[['x0', 'x1']].values, len(pipeline.predictor.model_hist))
        y_in = np.array_split(test_df[['y_true']].values, len(pipeline.predictor.model_hist))

        y_true = np.array_split(test_df['y_true'], len(pipeline.predictor.model_hist))
        y_pred = np.array_split(test_df['y_pred'], len(pipeline.predictor.model_hist))

        self.cost = train_df['y_true'].astype(float)*gain_factor
        self.cost = self.cost*np.random.normal(loc=2, size=len(train_df))
        self.cost = self.cost - (train_df['y_true'] == 0).astype(float)
        self.cost = self.cost.values.cumsum()
        self.final_cost = self.cost[-1]

        if trans is not None:
            gridt = trans.transform(self.get_grid())
        else:
            gridt = self.get_grid()

        ind = 0

        for i, mod in enumerate(pipeline.predictor.model_hist):
            if i % plot_step == 0:
                new_model = clone(pipeline.model)
                if hasattr(pipeline.model, 'coefs_'):
                    # print('Adding coefs_')
                    new_model.coefs_, new_model.intercepts_, new_model.classes_,\
                        new_model.n_outputs_, new_model.n_layers_, new_model.out_activation_,\
                        new_model._label_binarizer = mod[:-2]
                elif hasattr(pipeline.model, 'coef_'):
                    # print('Adding coef_')
                    new_model.coef_, new_model.intercept_, new_model.classes_ = mod[:-2]
                else:
                    print('Failed to add anything!')

                self.X.append(mod[-2])
                self.y.append(mod[-1].ravel())

                # self.cost.append(self.cost[-1] - len(np.hstack(self.y[-1]))+((np.hstack(self.y[-1]) == 1).sum()*(
                #             np.random.normal()+1)*gain_factor))

                if trans is not None:
                    Xt = trans.transform(np.vstack(self.X)[-self.window:, :])
                else:
                    Xt = np.vstack(self.X)[-self.window:, :]

                try:
                    Z = new_model.predict_proba(gridt)[:, 0].reshape(250, 250)
                    yPred = new_model.predict(Xt)
                except AttributeError:
                    Z = new_model.decision_function(gridt).reshape(250, 250)
                    yPred = new_model.predict(Xt)

                plt.subplot(121)
                cpl = plt.contourf(np.linspace(0, 1, 250),
                                   np.linspace(0, 1, 250),
                                   Z, 30, cmap='RdBu_r', vmin=0.2, vmax=0.8)


                # plt.colorbar(mappable=cpl, ax=ax, boundaries=np.arange(0, 1, .1))
                plt.contour(np.linspace(0, 1, 250),
                            np.linspace(0, 1, 250),
                            Z, [0.5])
                ax = plt.gca()
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(
                    ax_cb, cmap='RdBu_r',
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=1), orientation='vertical')
                plt.gcf().add_axes(ax_cb)

                # -- plot unlabbeled points
                ax.scatter(*X_in[i].T, c='k', s=2, alpha=.8)
                ax.scatter(*X_in[i][y_in[i].ravel() == 0].T, c='g', s=3, alpha=.5)
                ax.scatter(*X_in[i][y_in[i].ravel() == 1].T, c='b', s=3, alpha=.3)

                # -- plot labelled points
                ax.scatter(*np.vstack(self.X)[-self.window:, :].T,
                           c=self.cols[np.hstack(self.y)[-self.window:].astype(int)], alpha=.8)
                ax.scatter(*np.vstack(self.X)[-self.window:, :][np.hstack(self.y)[-self.window:] != yPred[-self.window:]].T,
                           c='r', s=3, alpha=.8)

                legend_elements = [matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                                   markerfacecolor='g', markersize=6, label='Not Fraud'),
                                   matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                                   markerfacecolor='b', markersize=6, label='Fraud'),
                                   matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                                   markerfacecolor='k', markersize=6, label='Unlabelled'), ]

                ax.legend(handles=legend_elements, loc='lower right')
                ind += len(mod[-2])
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                ax.set_title('Epoch: '+str(i), fontsize=15)

                self.acc.append(np.hstack(self.y)[-self.window:] == yPred[-self.window:])
                self.acc2.append(np.hstack(acc2[i]))

                plt.subplot(122)
                acc_plot = pd.Series(np.hstack(self.acc)).rolling(self.window).mean().values
                acc2_plot = pd.Series(np.hstack(self.acc2)).rolling(self.window).mean().values
                plt.plot(np.linspace(0, len(acc_plot), num=len(acc_plot)), acc_plot, c='m',
                         label='observed accuracy', alpha=.7, lw=2)
                plt.plot(np.linspace(0, len(acc_plot), num=len(acc2_plot)), acc2_plot, c='r',
                         label='actual accuracy', alpha=.6, lw=2)

                plt.ylim([0, 1.05])
                plt.xlim([0, len(np.hstack(pipeline.predictor.model_hist))])
                plt.title(f'Accuracy: {(acc_plot[-1]*100).round(2)}%',
                          fontsize=15)

                leg_elements = [matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                                markerfacecolor='m', markersize=6, label='Observerd Accuracy'), 
                                matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                                markerfacecolor='r', markersize=6, label='Validation set Accuracy'), ]
                if plot_cost:
                    ax1 = plt.gca()
                    divider = make_axes_locatable(ax1)
                    ax2 = divider.new_horizontal(size="20%", pad=0.05)
    #                 ax2.plot(np.linspace(0, len(np.hstack(acc)), num=len(cost)), cost, c='b', label='Profit')
                    # clrs = sns.hls_palette(2)
                    if self.cost[i] > 0:
                        ax2.bar(x=1, height=self.cost[i], width=4, color='g')
                    if self.cost[i] < 0:
                        ax2.bar(x=1, height=self.cost[i], width=4, color='r')
                    ax2.set_xlim(0, 2)
                    ax2.set_ylim(self.cost.min()+self.cost.min()*.3,
                                 self.cost.max()+self.cost.max()*.3)
                    ax2.set_title(f'Profit: ${self.cost[i].round(2)}')
                    ax2.axhline(c='k', linestyle='dashed')
                    ax2.tick_params(axis='y', bottom=False, right=True, left=False, top=False,
                                    labelbottom=False, labelright=True, labelleft=False)
                    plt.gcf().add_axes(ax2)
                    # ax.

                    # ax2.plot(np.linspace(0, len(np.hstack(self.acc)), num=len(self.cost)),
                    #          self.cost, c='c', alpha=.3, lw=4)
                    # ax2.scatter(np.linspace(0, len(np.hstack(self.acc)), num=len(self.cost)), self.cost,
                    #             c=np.array(clrs)[(np.array(self.cost) > 0).astype(int)], s=4, label='Profit')
                    # leg_elements.append(matplotlib.lines.Line2D([0], [0], c='w', marker='o', lw=0,
                    #                     markerfacecolor=clrs[1], markersize=6, label='Profit'),)

                plt.legend(handles=leg_elements, loc='lower right')

                f = plt.gcf()

                self.model_name = str(pipeline.model.__class__).split('.')[-1].split("'")[0]
                self.gener_name = gen_name
                if sup_title == '':
                    f.suptitle(f"Model: {self.model_name}\nDataSet: {self.gener_name}")
                else:
                    f.suptitle(sup_title, fontsize=20)

                if save_gif:
                    if not os.path.exists('tmp/'):
                        os.mkdir('tmp/')
                    self.filenames.append(f'tmp/gif{self.model_name}_{self.gener_name}_{i}.png')
                    f.savefig(self.filenames[-1])

                display.clear_output(wait=True)
                plt.show()

        if save_gif:
            self._save_gif(self.filenames, 'tmp/')
            # images = []

            # for filename in self.filenames:
            #     images.append(imageio.imread(filename))
            # imageio.mimsave(f'tmp/{model_name}_{gener_name}.gif', images)

            # if os.path.exists(f'tmp/{model_name}_{gener_name}.gif'):
            #     for filename in self.filenames:
            #           os.remove(filename)

    def performance(self, eval_data, train_data, window):
        """Plot the performance metrics of the model"""
        return performance(eval_data, train_data, window)

    def _save_gif(self, filenames, path='tmp/', ):
        """Combines output .png as a gif.

        Parameters
        ----------
        filenames: list
            list of filenames including the path to the files. i.e. tmp/image1.png

        path: string
            path to save the gif to
        """
        images = []
        for filename in self.filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{path}{self.model_name}_{self.gener_name}.gif', images)

        if os.path.exists(f'{path}{self.model_name}_{self.gener_name}.gif'):
            for filename in self.filenames:
                os.remove(filename)
