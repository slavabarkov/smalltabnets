from typing import Any

from sklearn.cross_decomposition import PLSRegression

from .base import BaseTabularRegressor


class PLSR(BaseTabularRegressor):
    """
    Partial Least-Squares Regression (PLSR) wrapper.
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        **kwargs: Any,  # Forward all common/base parameters
    ):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

        super().__init__(**kwargs)

    def _create_model(self, n_features: int):
        # Cap n_components at n_features to avoid errors
        # when the hyperparameter search proposes too many components.
        return PLSRegression(
            n_components=min(self.n_components, n_features),
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _fit_model(self, X, y, eval_set=None):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.model.fit(X, y)
        self.best_iteration = 0

    def _predict_model(self, X):
        y_pred = self.model.predict(X)
        return y_pred.squeeze(-1)
