from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from .base import BaseTabularRegressor


class RandomForestRegressor(BaseTabularRegressor):
    """Random Forest regressor with unified interface."""

    def _get_expected_params(self):
        return [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "criterion",
            "random_state",
            "n_jobs",
        ]

    def _create_model(self, n_features):
        params = self._get_model_params()

        # RandomForest doesn't support early stopping
        params.pop("early_stopping_rounds", None)
        print("Creating RandomForestRegressor with params:", params)
        return SklearnRandomForestRegressor(**params)

    def _fit_model(self, X, y, eval_set=None):
        # Random Forest doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs  # Always uses all trees
