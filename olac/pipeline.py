from queue import Queue


class QueuePoint():
    def __init__(self, point, index, predicted_label=None,
                 prob=None, true_label=None):
        self.point = point
        self.index = index
        self.predicted_label = predicted_label
        self.prob = prob
        self.true_label = true_label


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



