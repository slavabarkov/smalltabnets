from typing import Optional

from tabpfn import TabPFNRegressor as PriorLabsTabPFNRegressor

from .base import BaseTabularRegressor


class TabPFNRegressor(BaseTabularRegressor):
    """TabPFN regressor with unified interface."""

    def __init__(
        self,
        # Base parameters
        epochs=32,
        learning_rate=0.2,
        batch_size=256,
        use_early_stopping=False,
        early_stopping_rounds=None,
        # Dimensionality reduction
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # Misc
        device=None,
        random_state=42,
        verbose=0,
        **kwargs
    ):
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            random_state=random_state,
            verbose=verbose,
            use_pca=use_pca,
            n_pca_components=n_pca_components,
            **kwargs,
        )

    def _get_expected_params(self):
        return [
            "n_estimators",
            "random_state",
        ]

    def _create_model(self, n_features):
        params = self._get_model_params()
        return PriorLabsTabPFNRegressor(**params)

    def _fit_model(self, X, y, eval_set=None):
        # TabPFN doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs
