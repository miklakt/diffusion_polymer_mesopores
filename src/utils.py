import numpy as np
def mirror_extend(arr, axis=0):
    """
    Mirror (reflect) a numpy array along one axis and keep original,
    so the domain doubles along that axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array (any number of dimensions).
    axis : int
        Axis along which to mirror.

    Returns
    -------
    np.ndarray
        Extended array with shape doubled along the given axis.
    """
    mirrored = np.flip(arr, axis=axis)
    extended = np.concatenate([mirrored, arr], axis=axis)
    return extended