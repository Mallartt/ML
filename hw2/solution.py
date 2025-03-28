import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._coef = None
        self._intercept = None

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)
        else:
            return 0

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_features = x.shape
        random_array = np.random.default_rng(self.random_state)
        self._coef = random_array.normal(scale=0.01, size=n_features)
        self._intercept = np.array([0.0])
        best_loss = np.inf
        count_no_iter_change = 0
        split = int((1 - self.validation_fraction) * n_samples)
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
        for era in range(self.max_iter):
            if self.shuffle:
                index = random_array.permutation(x_train.shape[0])
                x_train = x_train[index]
                y_train = y_train[index]
            for i in range(0, x_train.shape[0], self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                y_pred = x_batch @ self._coef + self._intercept
                difference = y_pred - y_batch
                grad_coef = (x_batch.T @ difference) / len(x_batch)
                grad_coef += self.get_penalty_grad()
                grad_intercept = np.mean(difference)
                self._coef -= self.eta0 * grad_coef
                self._intercept -= self.eta0 * grad_intercept
            if self.early_stopping:
                y_test_pred = x_test @ self._coef + self._intercept
                loss = np.mean((y_test_pred - y_test) ** 2)
                if loss < best_loss - self.tol:
                    best_loss = loss
                    count_no_iter_change = 0
                else:
                    count_no_iter_change += 1
                    if count_no_iter_change == self.n_iter_no_change:
                        break
        return self

    def predict(self, x):
        return x @ self._coef + self._intercept

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
