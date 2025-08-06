import warnings
from typing import Literal, Optional

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge

from .base import BaseTabularRegressor


class LinearRegression(BaseTabularRegressor):
    """Linear Regression with regularization options and unified interface."""

    def __init__(
        self,
        *,
        regression_type: Literal["linear", "ridge", "lasso"] = "linear",
        alpha: float = 1.0,  # Regularization strength
        fit_intercept: bool = True,
        n_jobs: Optional[int] = None,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        self.regression_type = regression_type
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features):
        if self.regression_type == "linear":
            # Standard linear regression
            return SklearnLinearRegression(
                fit_intercept=self.fit_intercept,
                n_jobs=self.n_jobs,
            )
        elif self.regression_type == "ridge":
            # Ridge regression (L2 regularization)
            return Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
        elif self.regression_type == "lasso":
            # Suppress convergence warnings
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            # Lasso regression (L1 regularization)
            return Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown regression_type: {self.regression_type}")

    def _fit_model(self, X, y, eval_set=None):
        self.model.fit(X, y)
        self.best_iteration = 0
