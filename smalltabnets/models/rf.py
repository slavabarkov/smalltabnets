from typing import Optional

from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from .base import BaseTabularRegressor


class RandomForestRegressor(BaseTabularRegressor):
    """Random Forest regressor with unified interface."""

    def __init__(
        self,
        *,
        # RF specific parameters
        n_estimators: int = 1000,
        bootstrap: bool = True,
        criterion: str = "squared_error",
        n_jobs: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        kwargs.setdefault("epochs", self.n_estimators)

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features):
        return SklearnRandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def _fit_model(self, X, y, eval_set=None):
        # Random Forest doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs  # Always uses all trees
