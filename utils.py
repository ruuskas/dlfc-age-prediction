""""Helper functions"""

import numpy as np


def min_max_scale(arr, min_val=0, max_val=1):
    """Min-max scale an array to a specified range.
    Parameters:
    ----------
    arr : np.ndarray
        Input array to be scaled.
    min_val : float, optional
        Minimum value of the scaled range (default is 0).
    max_val : float, optional
        Maximum value of the scaled range (default is 1).
    Returns:
    -------
    np.ndarray
        Scaled array with values in the specified range.
    """
    nonzero_mask = ~np.isclose(arr, 0)
    arr_nonzero = arr[nonzero_mask]
    arr_min = np.min(arr_nonzero)
    arr_max = np.max(arr_nonzero)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr) + min_val
    scaled_arr = ((arr_nonzero - arr_min) / (arr_max - arr_min)) * (max_val - min_val) + min_val
    arr[nonzero_mask] = scaled_arr
    return arr
