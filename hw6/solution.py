import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        prediction = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            error = y - prediction
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(x, error)
            update = tree.predict(x)
            prediction += self.learning_rate * update
            self._estimators.append(tree)

    def predict(self, x):
        x = np.array(x)
        prediction = np.zeros(x.shape[0])
        for trees in self._estimators:
            prediction += self.learning_rate * trees.predict(x)
        return prediction

    @property
    def estimators_(self):
        return self._estimators


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.base_logit = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.base_logit = np.log(p / (1 - p))
        prediction = np.full_like(y, self.base_logit, dtype=float)
        for i in range(self.n_estimators):
            prob = 1 / (1 + np.exp(-prediction))
            error = y - prob
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(x, error)
            update = tree.predict(x)
            prediction += self.learning_rate * update
            self._estimators.append(tree)

    def predict_proba(self, x):
        x = np.array(x)
        prediction = np.full(x.shape[0], self.base_logit)
        for tree in self._estimators:
            prediction += self.learning_rate * tree.predict(x)
        prob = 1 / (1 + np.exp(-prediction))
        return np.vstack([1 - prob, prob]).T

    def predict(self, x):
        prob = self.predict_proba(x)[:, 1]
        return (prob >= 0.5).astype(int)

    @property
    def estimators_(self):
        return self._estimators
