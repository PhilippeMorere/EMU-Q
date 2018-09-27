import numpy as np


class FourierBasisFeatures:
    def __init__(self, dim, featureOrder, verbose=1):
        """
        Fourier basis features up to order N
        :param dim: input dimension
        :param featureOrder: Fourier feature order
        """

        freqs = tuple([list(range(featureOrder + 1))] * dim)
        # Cartesian product of arrays:
        prod = np.array(np.meshgrid(*freqs)).T.reshape(-1, dim)
        self.featureCoeff = np.pi * prod
        self.nFeatures = len(self.featureCoeff)
        if verbose >= 2:
            print("Feature coefficients({}): \n{}".format(
                len(self.featureCoeff), self.featureCoeff))

    def toFeatures(self, s):
        return np.cos(self.featureCoeff.dot(s.T)).T
