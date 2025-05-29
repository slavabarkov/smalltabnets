from typing import Optional

import xgboost as xgb

from .base import BaseTabularRegressor


class XGBRegressor(BaseTabularRegressor):
    """XGBoost regressor with unified interface."""

    def __init__(
        self,
        *,
        # XGBoost specific parameters
        n_estimators: int = 1000,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: Optional[int] = None,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs

        kwargs.setdefault("epochs", self.n_estimators)

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features):
        # XGBoost doesn't use explicit 'use_early_stopping' as a parameter
        # but handles it implicitly when 'early_stopping_rounds' is provided
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            verbosity=self.verbose,
            random_state=self.random_state,
        )

    def _fit_model(self, X, y, eval_set=None):
        # XGBoost handles early stopping internally
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=self.model.verbosity > 0,
        )

        # Set best iteration
        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration
        else:
            self.best_iteration = self.epochs
