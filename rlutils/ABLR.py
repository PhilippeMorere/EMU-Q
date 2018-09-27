import numpy as np

from rlutils.utils import smw_inv_correction
from rlutils.utils import batch_generator


class ABLR:
    """
    Purely analytic BLR with rank-k streaming updates
    """

    def __init__(self, M, alpha=1.0, beta=1.0, dOut=1, computeSninv=True):
        """
        Initialize model.
        :param M: Number of weights
        :param alpha: Weight prior precision
        :param beta: Noise precision
        :param dOut: Number of output dimensions
        :param computeSninv: Whether to compute the inverse of Sn. Useful for
        NLML computations (default=True).
        """
        self.N_trn = None  # The total number of training samples
        self.M = M
        self.dOut = dOut
        self.computeSninv = computeSninv

        # dimensionality
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self):
        self.Sninv_tgt = np.zeros(shape=(self.M, self.dOut))
        self.mn = np.zeros(shape=(self.M, self.dOut))
        self.tgt0 = np.zeros(shape=(self.M, self.dOut))
        self.Sn = np.identity(self.M) / self.alpha  # a.k.a. Ainv
        if self.computeSninv:
            self.Sninv = np.identity(self.M) * self.alpha  # a.k.a. A
        self.needRecomputeMean = True

    def update(self, phi, y):
        """
        Update BLR model with one a set of data points, performing a rank-k
        sherman-morisson-woodburry update.
        :param phi: Feature map for new points
        :param y: Target value for new points
        """

        self.Sn -= smw_inv_correction(A_inv=self.Sn,
                                      U=np.sqrt(self.beta) * phi.T,
                                      V=np.sqrt(self.beta) * phi)

        self.Sninv_tgt += self.beta * np.dot(phi.T, y)
        if self.computeSninv:
            self.Sninv += self.beta * np.dot(phi.T, phi)  # /
        self.needRecomputeMean = True

    def learn_from_history(self, all_phi, all_y, batch_size=None):
        """
        Train model on dataset by cutting it into batches for faster learning.
        :param all_phi: data features
        :param all_y: data targets
        :param batch_size: size of batches data set is cut into. Set to None
        for automatically computing optimal batch size (default=None).
        """
        # Define the batch data generator. This maintains an internal counter
        # and also allows wraparound for multiple epochs

        # Compute optimal batch size
        if batch_size is None:
            batch_size = int(np.cbrt(self.M ** 2 / 2))

        data_batch_gen = batch_generator(arrays=[all_phi, all_y],
                                         batch_size=batch_size,
                                         wrapLastBatch=False)

        N = all_phi.shape[0]  # Alias for the total number of training samples
        n_batches = int(np.ceil(N / batch_size))  # The number of batches

        """ Run the batched inference """
        for _ in range(n_batches):
            phi_batch, Y_batch = next(data_batch_gen)
            self.update(phi=phi_batch, y=Y_batch)

    def _recompute(self):
        self.mn = np.dot(self.Sn, self.Sninv_tgt)
        self.needRecomputeMean = False

    def predictMean(self, phi):
        """
        Model predictive mean.
        :param phi: Feature map for test data point
        :returns: predictive mean values for each test data point
        """
        if self.needRecomputeMean:
            self._recompute()
        Y_pred = np.dot(phi, self.mn)
        return np.atleast_2d(Y_pred)

    def predictVar(self, phi, includeBetaVar=True):
        """
        Model predictive variance.
        :param phi: Feature map for test data point
        :param includeBetaVar: Whether to include the 1/beta offset in variance
        :returns: predictive mean variances for each test data point
        """
        var = np.sum(np.dot(phi, self.Sn) * phi, axis=1, keepdims=True)
        if includeBetaVar:
            var += 1.0/self.beta
        return var

    def predict(self, phi):
        return self.predictMean(phi), self.predictVar(phi)

    def updateTargets(self, all_phi, all_t):
        """
        Update target for all datapoints.
        :param all_phi: Feature map for all data points
        :param all_t: Targets for all data points
        """
        self.Sninv_tgt = self.tgt0 + self.beta * np.dot(all_phi.T, all_t)
        self.needRecomputeMean = True


class fixedMeanABLR(ABLR):
    """ Variant of BLR, where the predictive mean returns to a fixed value
    away from data points. This uses the predictive variance, minimum and
    maximum variance to interpolate between true predictive mean and fixed
    mean.
    Note: While the value for maximum variance is correct, the one for
    minimum variance is only an empirical estimate, and depends on the feature
    map used. This implementation works for RFF.
    """
    def __init__(self, M, alpha, beta, fixedMean, sigRFF):
        super().__init__(M, alpha, beta)

        self.maxVar = 1.0 / alpha  # Not including 1.0/beta in computation
        # This is an empirical observation
        self.minVar = 1.0 / np.mean(sigRFF * (8*beta+alpha))
        self.fixedMean = fixedMean

    def predictMean(self, x_tst):
        mean = super().predictMean(x_tst)
        varRatio = (self.predictVar(x_tst, False) - self.minVar) / \
            (self.maxVar - self.minVar)
        varRatio[varRatio < 0] = 0
        return mean * (1-varRatio) + self.fixedMean * varRatio
