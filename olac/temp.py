from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.4"
__license__ = "MIT"


def detect_cusum(x, threshold=2, drift=0):
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

    self.pos_csum = [0]
    self.neg_csum = [0]

    self.alarm_index = []  # alarm time (index of when the change was detected)
    self.change_index = []  # index of when the change started
    self.pos_alarm_index = 0
    self.neg_alarm_index = 0

    self.alarm = False

    s = self.kalman_estimates[-1] - self.kalman_estimates[-2]
    self.pos_csum.append(self.pos_csum[-1] + s - drift)  # cumulative sum for + change
    self.neg_csum.append(self.neg_csum[-1] - s - drift)  # cumulative sum for - change

    if self.pos_csum[-1] < 0:
        self.pos_csum[-1] = 0
        self.pos_alarm_index = self.data_id

    if self.neg_csum[-1] < 0:
        self.neg_csum[-1] = 0
        self.neg_alarm_index = self.data_id

    if self.pos_csum[-1] > threshold or self.neg_csum[-1] > threshold:  # change detected!

        self.alarm = True
        self.alarm_index.append(self.data_id)    # alarm index

        if self.pos_csum[-1] > threshold:
            self.change_index.append(self.pos_alarm_index)  # start
        else:
            self.change_index.append(self.neg_alarm_index)  # start

        self.pos_csum[-1], self.neg_csum[-1] = 0, 0      # reset alarm
