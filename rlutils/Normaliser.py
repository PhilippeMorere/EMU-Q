import numpy as np


class Normaliser:
    """
    Normalise/unnormalise data to [0,1] or [-1,1].
    """
    def __init__(self, low, high, zeroOneInterval=True):
        """
        :param low: List of lower-bounds for each dimension
        :param high: List of upper-bounds for each dimension
        :param zeroOneInterval: whether normalised interval should be [0,1]
        (default) or [-1,1]
        """
        assert(len(low) == len(high) and
               "Upper and lower bounds much be same dimension.")
        assert(np.isfinite(np.sum(low)) and
               "Lower bound elements must be numbers.")
        assert(np.isfinite(np.sum(high)) and
               "Upper bound elements must be numbers.")

        spaceRange = np.array(high) - np.array(low)

        if np.sum(spaceRange > 100) > 0:
            print("Warning: normalising over large space.")

        self.factor = (1.0 if zeroOneInterval else 2.0) * spaceRange
        self.invFactor = (1.0 if zeroOneInterval else 2.0) / spaceRange
        self.offset = -np.array(low)
        self.finalOffset = 0.0 if zeroOneInterval else -1.0
        self.boundsNorm = (spaceRange * 0 - (0 if zeroOneInterval else 1),
                           spaceRange * 0 + 1)
        self.boundsOrig = (np.array(low), np.array(high))

    def normalise(self, x):
        """
        Normalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        _x = np.array(x)
        if len(_x.shape) == 1:
            assert(_x.shape == self.offset.shape and
                   "Data must be same dimension as lower/upper bounds")
        else:
            assert(_x.shape[1] == self.offset.shape[0] and
                   "Data must be same dimension as lower/upper bounds")

        return (_x + self.offset) * self.invFactor + self.finalOffset

    def unnormalise(self, x):
        """
        Unnormalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        _x = np.array(x)
        if len(_x.shape) == 1:
            assert(_x.shape == self.offset.shape and
                   "Data must be same dimension as lower/upper bounds")
        else:
            assert(_x.shape[1] == self.offset.shape[0] and
                   "Data must be same dimension as lower/upper bounds")

        return (_x - self.finalOffset) * self.factor - self.offset

    def boundsNormalised(self):
        return self.boundsNorm

    def boundsOriginal(self):
        return self.boundsOrig


if __name__ == "__main__":
    nrm = Normaliser([5, -10], [6, 100], True)

    # Test for single element in list
    x = [5.5, 4]
    y = nrm.normalise(x)
    z = nrm.unnormalise(y)
    print(x, y, z)
    assert(np.isclose(0, np.linalg.norm(x-z)))

    # Test for numpy array of elements
    x = np.hstack((np.arange(5, 6, 0.1).reshape(-1, 1),
                   np.arange(-10, 100, 11).reshape(-1, 1)))
    y = nrm.normalise(x)
    z = nrm.unnormalise(y)
    assert(np.isclose(0, np.linalg.norm(x-z)))
