import multiprocessing.dummy as mp
import pandas as pd
import numpy as np
import sklearn
import re
import pprint
from . import cost_of_label

from . import utils

from queue import Queue


def err_handler(e: Exception):
    """
    A simple error handler for threads that raises any Exceptions that occur.

    Parameters
    ----------
    e: The Exception being raised

    """
    raise e


class QueuePoint():
    """A data class for queued data points"""
    def __init__(self, point, index, predicted_label=None,
                 prob=None, true_label=None):
        self.point = point
        self.index = index
        self.predicted_label = predicted_label
        self.prob = prob
        self.true_label = true_label

    def to_tuple(self):
        """Convert the QueuePoint to a tuple.

        Returns
        -------
        tuple: (point, index, predicted_label, prob, true_label,)

        See also: olac.utils.queue_point_list_to_df

        """
        return (self.point, self.index, self.predicted_label,
                self.prob, self.true_label)


class BatchQueue(Queue):
    def get_all(self):
        """
        Get all items present in the queue at the time of calling and return
        them as a list, marking all the tasks as done immediately.

        Returns
        -------
        list: all current queue items

        """
        output = []
        for _ in range(self.qsize()):
            output.append(self.get())
            self.task_done()

        return output

    def put_all(self, items):
        """
        Put all the items in a list into the queue sequentially

        Parameters
        ----------
        items (iterable): The list of items to put

        """
        for i in items:
            self.put(i)


class Pipeline():
    """The pipeline runner class for backtesting online or dynamically
    retrained models.

    The pipeline gets data points from the data_generator and runs two
    worker threads in parallel, one for making predictions and retraining, and
    the other for buying labels. Their behaviour can be configured using
    the predictor and the labeller.

    The Pipeline also has properties which can be used by the labeller and the
    predictor for communicating with each other:
    * labelling_queue is the BatchQueue of points waiting to (potentially)
        be labelled
    * training_queue is the BatchQueue of labelled points available for
        model (re)training
    * stop_flag is set when the data_generator has been exhausted, and signals
        that the labelling worker should stop working and return its results.

    """
    def __init__(self, data_generator=None, model=None,
                 predictor=None, labeller=None,):
        """

        Parameters
        ----------
        data_generator: generator
            The data source. Should yeild datapoints of form [x0, x1, ..., y]

        model:
            The model to test. Should implement scikit-learn api.

        predictor: PredictorBase
            The predictor, which decides when to make predictions and
            fits the model. Should inherit from PredictorBase and implement
            methods train_condition, train_pipeline_model, and do_prediction

        labeller: LabellerBase
            The labeller, which decides which labels to buy. Should implement
            the methods buy_labels_condition and buy_labels.

        """

        self.data_generator = data_generator
        self.model = model
        self.predictor = predictor
        self.labeller = labeller

        # thread communication objects
        self.labelling_queue = BatchQueue()
        self.training_queue = BatchQueue()
        self._stop_flag = mp.Event()

    def run(self):
        """
        Run the pipeline until the data_generator is exhausted.

        Returns
        -------
        tuple: (training_set, eval_set)

        Training set is the list of QueuePoints that was labelled, and eval_set
        is the list of points that were not labelled.
        """
        for prop in [self.data_generator, self.model,
                     self.predictor, self.labeller]:
            assert prop is not None

        with mp.Pool(2) as pool:
            # run the labelling worker
            results = pool.apply_async(self._labelling_worker,
                                       error_callback=err_handler)

            # run the prediction worker
            pool.apply_async(self._prediction_worker,
                             error_callback=err_handler)

            # wait for the job to finish
            pool.close()
            pool.join()

        # collect the results
        return results.get()

    def _prediction_worker(self):
        # depends on the predictor
        assert self.predictor is not None

        # loop through the data_generator until exhausted
        for i, new_point in enumerate(self.data_generator):

            # check for training condition and train if met
            if self.predictor.train_condition(self):
                self.predictor.train_pipeline_model(self)

            x = new_point[:-1]
            y_true = new_point[-1]

            # get prediction from the predictor
            y_pred, prob = self.predictor.do_prediction(self, x)

            # place the point and prediction in the labelling_queue for the
            # labeller to process
            self.labelling_queue.put(QueuePoint(x, i, y_pred, prob, y_true))

        # signal the labelling worker to stop once no more data
        self._stop_flag.set()

    def _labelling_worker(self):
        # depends on the labeller
        assert self.labeller is not None

        # accumulators for labelled and unlabelled points
        training_set = []
        eval_set = []

        # run until signalled to stop
        while not self._stop_flag.is_set():
            if self.labeller.buy_labels_condition(self):

                labelled_points, unlabelled_points =\
                    self.labeller.buy_labels(self)

                # accumulate the results
                training_set += labelled_points
                eval_set += unlabelled_points

                # put all labelled points in the training_queue for processing
                # by the predictor
                self.training_queue.put_all(labelled_points)

        return training_set, eval_set

    def describe(pipeline):
        """
        Describe the pipeline
        """
        for thing in ['model', 'predictor', 'labeller', 'data_generator']:
            if thing == 'data_generator':
                print('='*20)
                print("{}:\n{}\n  {}\n".format(thing.capitalize(),
                                               '='*20,
                                               pipeline.__getattribute__(thing).__name__))

            else:
                print('='*20)
                print("{}:\n{}\n  {}\n".format(thing.capitalize(),
                                               '='*20,
                                               '\n  '.join(
                                        re.findall('(?<=\.)\w+',
                                                   str(pipeline.__getattribute__(thing).__class__)))))

                print('Parameters:  ')
                pprint.pprint(utils.get_params(pipeline.__getattribute__(thing)), indent=2)
                print()
        print('-'*80)


class PredictorBase():
    """Base class for Predictors for use with the Pipeline.

    No functionality is implemented here, this class should be subclassed and
    all methods of the subclass should accept the same arguments.

    The Pipeline argument to the methods gives access to the pipeline itself.
    Pipeline.training_queue contains the points that are available for training,
    and pipeline.model is the pipeline's model, which should implement the
    scikit-learn api.

    The Predictor may consume points from the training_queue, but putting
    predictions into the labelling queue is handled by the Pipeline. Predictions
    should simply be returned by do_prediction for further processing.

    See olac.pipeline.OnlinePredictor for an example.

    """
    def __init__(self, *args, **kwargs):
        """Use __init__ to set any parameters for the predictor."""
        pass

    def train_condition(self, pipeline,):
        """
        Called when a new data point is pulled from the data generator in the
        pipeline. This is the test for whether the model should be trained
        this iteration, before making the next prediction.

        If calculating the condition requires access to the datapoints
        themselves, make sure to store them in an object property as they will
        no longer be available in the queue for train_pipeline_model.

        Parameters
        ----------
        pipeline: the calling Pipeline object

        Returns
        -------
        bool: Whether to call self.train_pipeline_model this iteration

        """
        raise NotImplementedError(
            'Predictors must implement the train_condition method'
        )
        return True

    def train_pipeline_model(self, pipeline,):
        """
        Called by the Pipeline if self.train_condition is true. Should
        implement the logic for training pipeline.model (i.e. should call
        pipeline.model.fit or pipeline.model.partial_fit at some point).

        Parameters
        ----------
        pipeline: the calling Pipeline object

        Returns
        -------
        None

        """
        raise NotImplementedError(
            'Predictors must implement the train_pipeline_model method'
        )

    def do_prediction(self, pipeline, x,):
        """
        Use pipeline.model to make a prediction on point x. Should return
        the predicted class, as well as a measure of certainty.

        Parameters
        ----------
        pipeline: the calling Pipeline object
        x: the data point to predict

        Returns
        -------
        float, float : prediction, certainty
        """
        raise NotImplementedError(
            'Predictors must implement the do_prediction method'
        )

        y_pred, prob = 1, 0.5
        return y_pred, prob


class LabellerBase():
    """Base class for Labellers for use with the Pipeline.

    No functionality is implemented here, this class should be subclassed and
    all methods of the subclass should accept the same arguments.

    The Pipeline argument to the methods gives access to the pipeline itself.
    Pipeline.labelling_queue contains the points that are waiting for a
    decision on labelling.

    The Labeller may consume points from the labelling_queue, but putting
    predictions into the training queue is handled by the Pipeline. Lists of
    points to label should simply be returned by buy_labels for further
    processing.

    The labeller is stateful, so issues like e.g. tracking a budget for buying
    labels can be handled using properties of the object. The same is true for
    accumulating additional results.

    See olac.pipeline.ThresholdLabeller for an example.

    """
    def __init__(self, *args, **kwargs):
        """Use __init__ to set any parameters for the Labeller."""
        pass

    def buy_labels_condition(self, pipeline: Pipeline,):
        """
        Called continuously when not busy with labelling. Should return true
        when the condition is met for buying new labels.

        If calculating the condition requires access to the datapoints
        themselves, make sure to store them in an object property as they will
        no longer be available in the queue for buy_labels.

        Parameters
        ----------
        pipeline: the calling Pipeline object

        Returns
        -------
        bool: Whether to call self.buy_labels this iteration

        """
        raise NotImplementedError(
            'Labellers must implement the buy_labels_condition method'
        )
        return True

    def buy_labels(self, pipeline: Pipeline,):
        """
        Split the available points into labelled and unlabelled using some
        strategy. Should return two lists of QueuePoints: the list of points
        for which labels have been bought, and the remaining unlabelled points.

        Parameters
        ----------
        pipeline: the calling Pipeline object

        Returns
        -------
        list, list : labelled points, unlabelled points

        """
        raise NotImplementedError(
            'Labellers must implement the buy_labels method'
        )
        labelled_points, unlabelled_points = [], []
        return labelled_points, unlabelled_points


class OnlinePredictor(PredictorBase):
    """A simple predictor for use with online models. The corresponding
    pipeline.model should implement the partial_fit and the decision_function
    methods.

    """
    def __init__(self, verbose=True):
        """

        Parameters
        ----------
        verbose: bool
            Whether to print output when model is being updated
        """
        super().__init__()
        self.verbose = verbose

    def train_condition(self, pipeline,):
        """Train anytime there are points available in the training queue"""
        return not pipeline.training_queue.empty()

    def train_pipeline_model(self, pipeline,):
        """
        Update the model using all points available in the training queue
        using pipeline.model.partial_fit.
        """
        points = pipeline.training_queue.get_all()
        if self.verbose:
            print(
                f'Predictor:\t{len(points)} new points available, updating...'
            )

        X_train = np.vstack([np.array(p.point) for p in points])
        y_train = np.array([p.true_label for p in points])

        pipeline.model.partial_fit(X_train, y_train, classes=[0, 1])

    def do_prediction(self, pipeline, x,):
        """
        Make a prediction. If the model has not yet been fit (burn-in phase),
        return NaNs.
        """
        try:
            y_pred = pipeline.model.predict([x])

            # certainty measure not standard across sklearn
            # prefer predict_proba, else take decision_function
            if hasattr(pipeline.model, 'predict_proba'):
                prob = pipeline.model.predict_proba([x])
            elif hasattr(pipeline.model, 'decision_function'):
                prob = pipeline.model.decision_function([x])
            else:
                prob = np.nan

        except sklearn.exceptions.NotFittedError:
            # still burning in, return NaNs.
            y_pred = np.nan
            prob = np.nan

        return y_pred, prob


class ThresholdLabeller(LabellerBase):
    """A simple labeller. Once the number of points in the
    pipeline.labelling_queue reaches a certain threshold, all points are
    retrieved. Points are then randomly labelled with a certain probability.

    The total number of labels purchased is tracked using an internal
    property. This could be used for budgeting.

    """
    def __init__(self, threshold, prob, verbose=True):

        """

        Parameters
        ----------
        threshold: int
            The minimum number of points to trigger a batch of labelling

        prob: float (<= 1)
            The probability with which each point will receive a label

        verbose: bool
            Whether to print output when labelling
        """
        super().__init__()
        self.threshold = threshold
        self.prob = prob
        self.verbose = verbose

        self.labels_bought = 0

    def buy_labels_condition(self, pipeline: Pipeline,):
        """Buy labels if the labelling_queue is longer than the threshold."""
        n = pipeline.labelling_queue.qsize()
        if n > self.threshold:
            if self.verbose:
                print(
                    f'Labeller:\tThreshold met, {n} new '
                    'points available in queue'
                )
            return True
        else:
            return False

    def buy_labels(self, pipeline: Pipeline,):
        """Get all the points from the labelling queue and label them with
        some probability. """
        labelled_points = []
        unlabelled_points = []

        points = pipeline.labelling_queue.get_all()

        for point in points:
            # self.prob percent chance of being labelled
            if np.random.uniform(0, 1) < self.prob:
                self.labels_bought += 1
                labelled_points.append(point)
            else:
                unlabelled_points.append(point)

        if self.verbose:
            print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points


class NaieveLabeller(LabellerBase):
    """ A Naieve labeller which decides on how to get the labels randomly. Also is the function to determine the
    cost of investigating a certain datapoint implemented. The costs per investigation round are kept track of in the
    inherited list : cost_of_points.

    """

    def __init__(self, threshold, decider, decider_args=tuple(), decider_kwargs=dict()):
        """

        Parameters
        ----------
        threshold: The minimum number of points to trigger a batch of labelling
        decider: the function that decides wheter to buy or not to buy a label of a certain point
        decider_args: args of decider function
        decider_kwargs: kwargs of decider function
        """
        super().__init__()
        self.threshold = threshold
        self.decider = decider
        self.decider_args = decider_args
        self.decider_kwargs = decider_kwargs
        self.labels_bought = 0
        self.cost_points = []

    def buy_labels_condition(self, pipeline, ):
        """Buy labels if the labelling_queue is longer than the threshold."""
        n = pipeline.labelling_queue.qsize()
        if n > self.threshold:
            print(f'Labeller:\tThreshold met, {n} new points avaible in queue')
            return True
        else:
            return False

    def buy_labels(self, pipeline, ):
        """Get all the points from the labelling queue and label them according to the decissions made in the decider
         function. Optional is to replace the decider function with a more advanced decider function. The function

          Returns
          -------
          array of labelled_points, unlabelled_points"""
        labelled_points = []
        unlabelled_points = []

        # Get all the points from the queue (queue will now be empty)
        points = pipeline.labelling_queue.get_all()

        # use the decider function to create a list of which points to investigate an which not
        decider = self.decider(self, points, *self.decider_args, **self.decider_kwargs)

        # calculate the cost of the investigation
        cost = cost_of_label.cost_of_label(data=points, decision=decider, salary=-1.5, data_type='array')

        # store the data points
        for i, point in enumerate(points):
            if decider[i] == 1:
                labelled_points.append(point)
            else:
                unlabelled_points.append(point)

        self.cost_points += [cost.sum()]
        print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points


class GridPredictor():
    """
    Mixin class to add a grid predictor to your base predictor class. The
    class will add a get_grid and grid_predict method to your class,
    that can be used to save model predictions done on a 250 by 250 grid
    which can be used to plot the decision boundary.
    """
    def get_grid(self):
        gridpoints = np.linspace(-10, 10, 250)
        grid = []
        for y in gridpoints:
            for x in gridpoints:
                grid.append([x, y])
        return np.array(grid)

    def grid_predict(self, pipeline):
        l = int(np.sqrt(self.grid.shape[0]))
        return pipeline.model.predict_proba(self.grid)[:, 0].reshape(l, l)


class OfflinePredictor(GridPredictor, PredictorBase):
    """A simple predictor for use with non-online models. The corresponding
    pipeline.model should implement the partial_fit and the decision_function
    or predict_proba methods.

    """
    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.grid = self.get_grid()
        self.hist_grid = []
        self.historic_points = []

    def train_condition(self, pipeline, ):
        """Train anytime there are batch_size points in the queue"""
        return pipeline.training_queue.qsize() >= self.batch_size

    def train_pipeline_model(self, pipeline,):
        """
        Rertrain the model using all points available in the training queue
        plus all the points you've come across before using pipeline.model.fit.
        """
        # Get the new points from the queue
        new_points = pipeline.training_queue.get_all()  # training_queue is now empty
        print(f'Predictor:\t{len(new_points)} new points available, re-training...')

        # Stack the points as an array to add to historical points
        add_points = np.vstack([np.hstack((p.point, p.true_label)) for p in new_points])
        try:
            # Get the historical points
            self.historic_points = np.vstack((self.historic_points, add_points))
        except ValueError:
            # If no historic points yet, set first batch as historical points
            self.historic_points = add_points

        # Split into X and y
        X_train = self.historic_points[:, :-1]
        y_train = self.historic_points[:, -1]

        # Train the model
        pipeline.model.fit(X_train, y_train)
        print("Trained model on {} points".format(len(self.historic_points)))

    def do_prediction(self, pipeline, x,):
        """
        Make a prediction. If the model has not yet been fit (burn-in phase),
        return NaNs.
        """
        try:
            y_pred = pipeline.model.predict([x])
            self.hist_grid.append(self.grid_predict(pipeline))
            if hasattr(pipeline.model, 'predict_proba'):
                prob = pipeline.model.predict_proba([x])
            elif hasattr(pipeline.model, 'decision_function'):
                prob = pipeline.model.decision_function([x])
            else:
                prob = np.nan

        # Not trained yet, return nans
        except sklearn.exceptions.NotFittedError:
            y_pred = np.nan
            prob = np.nan

        return y_pred, prob

