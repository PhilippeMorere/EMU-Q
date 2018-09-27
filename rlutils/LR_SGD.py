import numpy as np


class BaseOptimizer(object):
    """Base (Stochastic) gradient descent optimizer
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    Attributes
    ----------
    learning_rate : float
        the current learning rate
    """

    def __init__(self, params, learning_rate_init=0.1):
        self.params = params
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, grads):
        """Update parameters with given gradients
        Parameters
        ----------
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        return self._get_updates(grads)

    def iteration_ends(self, time_step):
        """Perform update to learning rate and potentially other states at the
        end of an iteration
        """
        pass

    def trigger_stopping(self, msg, verbose):
        """Decides whether it is time to stop training
        Parameters
        ----------
        msg : str
            Message passed in for verbose output
        verbose : bool
            Print message to stdin if True
        Returns
        -------
        is_stopping : bool
            True if training needs to stop
        """
        if verbose:
            print(msg + " Stopping.")
        return True

    def reset(self):
        """Resets object.
        """
        pass


class ConstantRate(BaseOptimizer):
    """Constant learning rate for gradient descent
    parameters
    ---------
    learning_rate: float, optional, default 0.01
        The constant learning rate used.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """

        return self.learning_rate * grads


class SGDOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer with momentum
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default 'constant'
        Learning rate schedule for weight updates.
        -'constant', is a constant learning rate given by
         'learning_rate_init'.
        -'invscaling' gradually decreases the learning rate 'learning_rate_' at
          each time step 't' using an inverse scaling exponent of 'power_t'.
          learning_rate_ = learning_rate_init / pow(t, power_t)
        -'adaptive', keeps the learning rate constant to
         'learning_rate_init' as long as the training keeps decreasing.
         Each time 2 consecutive epochs fail to decrease the training loss by
         tol, or fail to increase validation score by tol if 'early_stopping'
         is on, the current learning rate is divided by 5.
    momentum : float, optional, default 0.9
        Value of momentum used, must be larger than or equal to 0
    nesterov : bool, optional, default True
        Whether to use nesterov's momentum or not. Use nesterov's if True
    Attributes
    ----------
    learning_rate : float
        the current learning rate
    velocities : list, length = len(params)
        velocities that are used to update params
    """

    def __init__(self, params, learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, nesterov=True, power_t=0.5):
        super(SGDOptimizer, self).__init__(params, learning_rate_init)

        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = np.zeros_like(params).reshape(-1, 1)

    def iteration_ends(self, time_step):
        """Perform updates to learning rate and potential other states at the
        end of an iteration
        Parameters
        ----------
        time_step : int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        """
        if self.lr_schedule == 'invscaling':
            self.learning_rate = (float(self.learning_rate_init) /
                                  (time_step + 1) ** self.power_t)

    def trigger_stopping(self, msg, verbose):
        if self.lr_schedule == 'adaptive':
            if self.learning_rate > 1e-6:
                self.learning_rate /= 5.
                if verbose:
                    print(msg + " Setting learning rate to %f" %
                          self.learning_rate)
                return False
            else:
                if verbose:
                    print(msg + " Learning rate too small. Stopping.")
                return True
        else:
            if verbose:
                print(msg + " Stopping.")
            return True

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        updates = self.momentum * self.velocities - self.learning_rate * grads
        self.velocities = updates

        if self.nesterov:
            updates = self.momentum * self.velocities \
              - self.learning_rate * grads

        return updates

    def reset(self):
        self.learning_rate = float(self.learning_rate_init)


class LR_SGD:
    def __init__(self, M, learningRate=ConstantRate(0.01), dOut=1):
        """
        :param M: Number of weights.
        :param learningRate: a opt.stochasticOptimiser.BaseOptimizer object
        :param dOut: Number of output dimensions.
        """
        self.M = M
        self.dOut = dOut
        self.learningRate = learningRate
        self.isModelInit = False
        self.reset()

    def reset(self):
        self.learningRate.reset()
        self.w = np.random.normal(0.0, 1.0, (self.M, self.dOut))
        self.time = 0

    def isInit(self):
        return self.isModelInit

    def update(self, phi, y):
        self.isModelInit = True
        grads = np.dot(phi.T, y - np.dot(phi, self.w))
        deltaw = self.learningRate.update_params(grads)
        self.w += deltaw.reshape(self.w.shape)

        if isinstance(self.learningRate, SGDOptimizer):
            self.time += 1
            self.learningRate.iteration_ends(self.time)

    def predictMean(self, phi):
        if self.isModelInit:
            y = np.dot(phi, self.w).reshape(-1, self.dOut)
            return y
        else:
            return np.zeros((phi.shape[0], self.dOut))

    def predict(self, phi):
        return self.predictMean, None

    def optimise(self, max_evals=200):
        # TODO optimise parameters
        pass
