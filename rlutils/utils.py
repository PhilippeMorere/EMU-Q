import numpy as np


def smw_inv_correction(A_inv, U, V):
    """
    Sherman-Morrison-Woodbury update
    For rank k updates to the inverse matrix

    IMPORTANT: This is the correction factor which one must subtract from A_inv
    Usage:   subtract this value from current A_inv

    ref:     http://mathworld.wolfram.com/WoodburyFormula.html
             https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    :param A_inv:   n x n
    :param U:      n x k
    :param V:      k x n
    :return:
    """
    rank = U.shape[1]
    SU = np.dot(A_inv, U)
    VS = np.dot(V, A_inv)
    I_plus_VSU_inv = np.linalg.pinv(np.identity(rank) + np.dot(VS, U))
    SU_I_plus_VSU = np.dot(SU, I_plus_VSU_inv)
    return np.dot(SU_I_plus_VSU, VS)


def batch_generator(arrays, batch_size, wrapLastBatch=False):
    """
    Batch generator() function for yielding [x_train, y_train] batch slices for
    numpy arrays
    Appropriately deals with looping back around to the start of the dataset
    Generate batches, one with respect to each array's first axis.
    :param arrays:[array, array]  or [array, None]...
                  e.g. [X_trn, Y_trn] where X_trn and Y_trn are ndarrays
    :param batch_size: batch size
    :param wrapLastBatch: whether the last batch should wrap around dataset
    to include first datapoints (True), or be smaller to stop at the end of
    the dataset (False).
    :return:
    """
    starts = [0] * len(
        arrays)  # pointers to where we are in iteration     --> [0, 0]
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                if wrapLastBatch:
                    batch = np.concatenate((array[start:], array[:diff]))
                    starts[i] = diff
                else:
                    batch = array[start:]
                    starts[i] = 0
            batches.append(batch)
        yield batches
