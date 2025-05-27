from tabpfn import TabPFNRegressor as PriorTabPFNRegressor

from .base import BaseTabularRegressor


class TabPFNRegressor(BaseTabularRegressor):
    """TabPFN regressor with unified interface."""

    def _get_expected_params(self):
        return [
            "n_estimators",
            "random_state",
        ]

    def _create_model(self, n_features):
        params = self._get_model_params()
        return PriorTabPFNRegressor(**params)

    def _fit_model(self, X, y, eval_set=None):
        # TabPFN doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs
