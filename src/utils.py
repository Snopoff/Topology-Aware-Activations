from errno import EEXIST
from os import makedirs, path
import numpy as np


def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line

    https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def obtain_points_for_each_label(X, labels):
    """
    Given a dataset X with the labels, splits data by them
    """
    res = dict()
    label_vals = np.unique(labels)
    for label in label_vals:
        label_mask = labels == label
        res[label] = X[label_mask, :]
    return res
