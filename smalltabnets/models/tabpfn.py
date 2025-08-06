from typing import Optional

from tabpfn import TabPFNRegressor as PriorLabsTabPFNRegressor
from tabpfn.config import ModelInterfaceConfig

from .base import BaseTabularRegressor


class TabPFNRegressor(BaseTabularRegressor):
    """TabPFN regressor with unified interface."""

    def __init__(
        self,
        *,
        # TabPFN specific parameters
        n_estimators: int = 32,
        ignore_pretraining_limits: bool = True,
        feature_shift_method: str = "shuffle",
        fingerprint_feature: bool = True,
        polynomial_features: str = "no",
        subsample_samples: Optional[int] = None,
        standardize_targets_tabpfn: bool = True,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.feature_shift_method = feature_shift_method
        self.fingerprint_feature = fingerprint_feature
        self.polynomial_features = polynomial_features
        self.subsample_samples = subsample_samples
        self.standardize_targets_tabpfn = standardize_targets_tabpfn

        self.inference_config = ModelInterfaceConfig(
            FEATURE_SHIFT_METHOD=self.feature_shift_method,
            FINGERPRINT_FEATURE=self.fingerprint_feature,
            POLYNOMIAL_FEATURES=self.polynomial_features,
            SUBSAMPLE_SAMPLES=self.subsample_samples,
            REGRESSION_Y_PREPROCESS_TRANSFORMS=(
                None,
                "safepower" if self.standardize_targets_tabpfn else None,
            ),
        )

        kwargs.setdefault("epochs", self.n_estimators)

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features):
        return PriorLabsTabPFNRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
            ignore_pretraining_limits=True,
            inference_config=self.inference_config,
        )

    def _fit_model(self, X, y, eval_set=None):
        # TabPFN doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs - 1
