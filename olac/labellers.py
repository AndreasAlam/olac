import numpy as np
from . import pipeline
from . import utils as ut


class CertaintyLabeller(pipeline.LabellerBase):
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

    def buy_labels_condition(self, pipeline: pipeline.Pipeline,):
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

    def buy_labels(self, pipeline: pipeline.Pipeline,):
        """Get all the points from the labelling queue and label them with
        some probability. """
        labelled_points = []
        unlabelled_points = []

        points = pipeline.labelling_queue.get_all()

        # -- filter out the nans
        list_points = [p.to_tuple() for p in points if type(p.to_tuple()[-2]) == np.ndarray]

        if len(list_points) == 0:
            list_points = [p.to_tuple() for p in points]
        else:
            list_points = sorted(list_points, key=lambda point: point[-2][0][1], reverse=True)

        try:
            n = int(len(list_points)/2)
            top = np.array(list_points)[:n,1]

#         top10 = np.array(top10)[:, 1]

            for point in points:
                # self.prob percent chance of being labelled
                if point.index in top:
                    self.labels_bought += 1
                    labelled_points.append(point)
                else:
                    unlabelled_points.append(point)
        except IndexError:
            print("No points added")

        if self.verbose:
            print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points


class GreedyLabeller(pipeline.ThresholdLabeller):
    def buy_labels(self, pipeline):

        def get_prob(point):
            p = point.prob
            try:
                return p.flatten()[-1]
            except:
                return p

        points = pipeline.labelling_queue.get_all()
        points = list(sorted(
            points, key=get_prob, reverse=True
        ))

        n_labs = int(np.round(self.prob*len(points)))
        n_labs = max(n_labs, 1)
        n_labs = min(n_labs, len(points))

        labelled_points = points[:n_labs]
        unlabelled_points = points[n_labs:]

        if self.verbose:
            print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points


class VeryGreedyLabeller(pipeline.ThresholdLabeller):
    def buy_labels(self, pipeline):

        def get_prob(point):
            p = point.prob
            try:
                return p.flatten()[-1]
            except:
                return p

        labelled_points = []
        unlabelled_points = []

        points = pipeline.labelling_queue.get_all()

        for point in points:
            prob = get_prob(point)

            if np.isnan(prob):
                labelled_points.append(point)
            elif prob >= 0.5:
                labelled_points.append(point)
            elif np.random.uniform() < self.prob:
                labelled_points.append(point)
            else:
                unlabelled_points.append(point)

        if self.verbose:
            print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points


class UncertaintyLabeller(pipeline.ThresholdLabeller):
    def buy_labels(self, pipeline):

        prob = hasattr(pipeline.model, 'predidct_proba')

        def uncertainty(point):
            try:
                p = point.prob.flatten()[-1]
            except:
                p = point.prob

            if prob:
                uncert = (p-0.5)**2
            else:
                uncert = p**2

            return uncert

        points = pipeline.labelling_queue.get_all()
        points = list(sorted(
            points, key=uncertainty, reverse=False
        ))

        n_labs = int(np.round(self.prob*len(points)))
        n_labs = max(n_labs, 1)
        n_labs = min(n_labs, len(points))

        labelled_points = points[:n_labs]
        unlabelled_points = points[n_labs:]

        if self.verbose:
            print(f'Labeller:\tLabelled {len(labelled_points)} new points')

        return labelled_points, unlabelled_points

