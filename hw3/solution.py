import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        pass

    def get_penalty_grad(self):
        pass

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        pass

    def predict(self, x):
        pass

    @staticmethod
    def softmax(z):
        """
        Calculates a softmax normalization over the last axis

        Examples:

        >>> softmax(np.array([1, 2, 3]))
        [0.09003057 0.24472847 0.66524096]

        >>> softmax(np.array([[1, 2, 3], [4, 5, 6]]))
        [[0.09003057 0.24472847 0.66524096]
         [0.03511903 0.70538451 0.25949646]]
        :param z: np.array, size: (d0, d1, ..., dn)
        :return: np.array of the same size as z
        """
        pass

    @property
    def coef_(self):
        pass

    @property
    def intercept_(self):
        pass

    @coef_.setter
    def coef_(self, value):
        pass

    @intercept_.setter
    def intercept_(self, value):
        pass
