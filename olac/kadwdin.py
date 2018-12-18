import numpy as np


class K_ADWIN(object):
    """

    Parameters
    ----------

    Attributes
    ----------
    labels_ : The cluster labels identified during fitting
    components_ : The vector array input used in fitting

    _dbscan : The internal DBSCAN classifier
    _meanshift : The internal Mean Shift classifier
    _knn : The internal KNN classifier
    """
    def __init__(self):
        pass


class Adwin:
    """Adapted from https://github.com/SmileYuhao/ConceptDrift
    """
    def __init__(self, delta=0.002, max_buckets=5, min_clock=32, min_length_window=10, min_length_sub_window=5):
        """
        :param delta: confidence value
        :param max_buckets: max number of buckets which have same number of original date in one row
        :param min_clock: min number of new data for starting to reduce window and detect change
        :param min_length_window: min window's length for starting to reduce window and detect change
        :param min_length_sub_window: min sub window's length for starting to reduce window and detect change
        """
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_clock = min_clock
        self.min_length_window = min_length_window
        self.min_length_sub_window = min_length_sub_window
        # time is used for comparison with min_clock parameter
        self.time = 0
        # width of the window
        self.width = 0
        # sum of all values in the window
        self.total = 0.0
        # incremental variance of all values in the window
        self.variance = 0.0
        # number of buckets that held the values
        # this value has the upper limit set by max_buckets
        self.bucket_number = 0
        # last_bucket_row: defines the max number of merged
        self.last_bucket_row = 0
        self.list_row_buckets = AdwinList(self.max_buckets)

    def set_input(self, value):
        """
        Main method for adding a new data value and automatically detect a possible concept drift.
        :param value: new data value
        :return: true if there is a concept drift, otherwise false
        """
        self.time += 1
        # Insert the new element
        self.__insert_element(value)
        # Reduce window
        return self.__reduce_window()

    def __insert_element(self, value):
        """
        Insert a new element by creating a new bucket for the head element of the list. The overall variance and
        total value are updated incrementally. At the end, buckets maybe compressed (merged) if the maximum number of
        buckets has been reached.
        :param value: new data value from the stream
        """
        self.width += 1
        # Insert the new bucket
        self.list_row_buckets.head.insert_bucket(value, 0)
        self.bucket_number += 1
        # Calculate the incremental variance
        incremental_variance = 0
        if self.width > 1:
            incremental_variance = (self.width - 1) * np.pow(2, (value - self.total / (self.width - 1))) / self.width
        self.variance += incremental_variance
        self.total += value
        # compress (merge) buckets if necessary
        self.__compress_buckets()

    def __compress_buckets(self):
        """Merging two buckets corresponds to creating a new bucket whose size is equal to the sum of the sizes of
        those two buckets. The size of a bucket means how many original data is contained inside it.
        """
        cursor = self.list_row_buckets.head
        i = 0
        while cursor is not None:
            # Find the number of buckets in a row
            k = cursor.bucket_size_row

            # Merge buckets if row is full
            if k == self.max_buckets + 1:
                next_node = cursor.next
                if next_node is None:
                    self.list_row_buckets.add_to_tail()
                    # new list item was added to the list
                    # hence, next pointer has been reset now to this new tail
                    next_node = cursor.next
                    self.last_bucket_row += 1

                n1 = np.pow(2, i)
                n2 = np.pow(2, i)

                # consider values from buckets 0 and 1 as these are the heading bucket elements inside a list item
                u1 = cursor.bucket_total[0] / n1
                u2 = cursor.bucket_total[1] / n2

                external_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)

                # create and insert a new bucket into the next list item
                new_bucket_total = cursor.bucket_total[0] + cursor.bucket_total[1]
                new_bucket_variance = cursor.bucket_variance[0] + cursor.bucket_variance[1] + external_variance
                next_node.insert_bucket(new_bucket_total, new_bucket_variance)
                self.bucket_number += 1

                # remove 2 buckets from the current list item
                cursor.compress_buckets_row(2)

                # stop if the the max number of buckets does not exceed for the next item list
                if next_node.bucket_size_row <= self.max_buckets:
                    break
            else:
                break
            cursor = cursor.next
            i += 1

    def __reduce_window(self):
        """
        Detect a change in the distribution and reduce the window if there is a concept drift.
        :return: boolean: whether has changed
        """
        is_changed = False
        if self.time % self.min_clock == 0 and self.width > self.min_length_window:
            is_reduced_width = True
            while is_reduced_width:
                is_reduced_width = False
                is_exit = False
                n0, n1 = 0, self.width
                u0, u1 = 0, self.total

                # start building sub windows from the tail (old entries)
                cursor = self.list_row_buckets.tail
                i = self.last_bucket_row
                while (not is_exit) and (cursor is not None):
                    for k in range(cursor.bucket_size_row):
                        # In case of n1 equals 0
                        if i == 0 and k == cursor.bucket_size_row - 1:
                            is_exit = True
                            break

                        # sub window 0 is growing while sub window 1 is getting smaller
                        n0 += pow(2, i)
                        n1 -= pow(2, i)
                        u0 += cursor.bucket_total[k]
                        u1 -= cursor.bucket_total[k]
                        diff_value = (u0 / n0) - (u1 / n1)

                        # remove old entries iff there is a concept drift and the minimum sub window length is matching
                        if n0 > self.min_length_sub_window + 1 and n1 > self.min_length_sub_window + 1 and \
                                self.__reduce_expression(n0, n1, diff_value):
                            is_reduced_width, is_changed = True, True
                            if self.width > 0:
                                n0 -= self.__delete_element()
                                is_exit = True
                                break
                    cursor = cursor.previous
                    i -= 1
        return is_changed

    def __reduce_expression(self, n0, n1, diff_value):
        """
        Calculate epsilon_cut value and check if difference between the mean values of two sub windows is greater than
        it.
        :param n0: number of elements in sub window 0
        :param n1: number of elements in sub window 1
        :param diff_value: difference of mean values of both sub windows
        :return: true if difference of mean values is higher than epsilon_cut
        """
        # harmonic mean of n0 and n1 (originally: 1 / (1/n0 + 1/n1))
        m = 1 / (n0 - self.min_length_sub_window + 1) + 1 / (n1 - self.min_length_sub_window + 1)
        d = np.log(2 * np.log(self.width) / self.delta)
        variance = self.variance / self.width
        epsilon = np.sqrt(2 * m * variance * d) + 2 / 3 * m * d
        return np.abs(diff_value) > epsilon

    def __delete_element(self):
        """
        Remove a bucket from tail of window
        :return: Number of elements to be deleted
        """
        # last list item (the oldest bucket) with the oldest entry at first internal array position
        node = self.list_row_buckets.tail
        deleted_number = np.pow(2, self.last_bucket_row)
        self.width -= deleted_number
        self.total -= node.bucket_total[0]
        deleted_element_mean = node.bucket_total[0] / deleted_number

        incremental_variance = (node.bucket_variance[0] + deleted_number * self.width *
                                np.pow(2, (deleted_element_mean - self.total / self.width)) /
                                (deleted_number + self.width))

        self.variance -= incremental_variance
        # Delete bucket
        node.compress_buckets_row(1)
        self.bucket_number -= 1
        # if after removing an entry, the bucket becomes empty, remove it from the tail
        if node.bucket_size_row == 0:
            self.list_row_buckets.remove_from_tail()
            self.last_bucket_row -= 1
        return deleted_number


class AdwinList:
    """Add new bucket at head of window (which has smaller number of merged),
    remove old bucket from tail of window (which has bigger number of merged)

    Adapted from https://github.com/SmileYuhao/ConceptDrift
    """
    def __init__(self, max_buckets=5):
        """
        :param max_buckets:max number of element in each bucket
        """
        self.count = 0
        self.max_buckets = max_buckets
        self.head = None
        self.tail = None

        # Add the object at the beginning of the window
        self.head = AdwinListItem(self.max_buckets, next=self.head)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        """Add the object at the end of the window
        """
        self.tail = AdwinListItem(self.max_buckets, previous=self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        """Remove the last object in the window
        """
        self.tail = self.tail.previous
        if self.tail is None:
            self.head = None
        else:
            self.tail.next = None
        self.count -= 1


class AdwinListItem:
    """
    A list item contains a list (array) of buckets limited to a maximum of buckets as set in '__init__'. As the new
    buckets are added at the end of the internal array, when old entries need to be removed, they are taken from the
    head of this array. Each item has a connection to a previous and next list item.

    Adapted from https://github.com/SmileYuhao/ConceptDrift
    """

    def __init__(self, max_buckets=5, next=None, previous=None):
        self.max_buckets = max_buckets
        # current number of buckets in this list item
        self.bucket_size_row = 0

        self.next = next
        # add the 'previous' connection of the following list item to this item
        if next is not None:
            next.previous = self

        self.previous = previous
        # add the 'next' connection of the previous list item to this item
        if previous is not None:
            previous.next = self

        self.bucket_total = np.zeros(self.max_buckets + 1)
        self.bucket_variance = np.zeros(self.max_buckets + 1)

    def insert_bucket(self, value, variance):
        """
        Insert a new bucket at the end of the array.
        """
        self.bucket_total[self.bucket_size_row] = value
        self.bucket_variance[self.bucket_size_row] = variance
        self.bucket_size_row += 1

    def compress_buckets_row(self, number_deleted):
        """
        Remove the 'number_deleted' buckets as they are the oldest ones.
        """
        delete_index = self.max_buckets - number_deleted + 1
        self.bucket_total[:delete_index] = self.bucket_total[number_deleted:]
        self.bucket_total[delete_index:] = np.zeros(number_deleted)

        self.bucket_variance[:delete_index] = self.bucket_variance[number_deleted:]
        self.bucket_variance[delete_index:] = np.zeros(number_deleted)

        self.bucket_size_row -= number_deleted


def detect_cusum(x, threshold=1, drift=0):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.

    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.

    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.

    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.

    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.

    See this IPython Notebook [2]_.

    References
    ----------
    .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
    .. [2] hhttp://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

    Examples
    --------
    >>> from detect_cusum import detect_cusum
    >>> x = np.random.randn(300)/5
    >>> x[100:200] += np.arange(0, 4, 4/100)
    >>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)

    >>> x = np.random.randn(300)
    >>> x[100:200] += 6
    >>> detect_cusum(x, 4, 1.5, True, True)

    >>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
    >>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
    """

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm

    return ta, tai


def aa(x, drift, threshold):
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm

    return ta, tai, taf, amp
