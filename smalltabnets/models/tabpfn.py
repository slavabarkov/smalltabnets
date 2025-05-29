from typing import Optional

from tabpfn import TabPFNRegressor as PriorLabsTabPFNRegressor
from tabpfn.config import ModelInterfaceConfig

from .base import BaseTabularRegressor


class TabPFNRegressor(BaseTabularRegressor):
    """TabPFN regressor with unified interface."""

    def __init__(
        self,
        # Base training parameters
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        # Base early stopping parameters
        use_early_stopping: Optional[bool] = None,
        early_stopping_rounds: Optional[int] = None,
        # Base preprocessing parameters
        feature_scaling: bool = None,
        standardize_targets: bool = False,
        clip_features: bool = False,
        clip_outputs: bool = False,
        # Base dimensionality reduction parameters
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # Base system and utility parameters
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        # TabPFN specific parameters
        n_estimators: int = 32,
        ignore_pretraining_limits: bool = True,
        feature_shift_method: str = "shuffle",
        class_shift_method: Optional[str] = "shuffle",
        fingerprint_feature: bool = True,
        polynomial_features: str = "no",
        subsample_samples: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.feature_shift_method = feature_shift_method
        self.class_shift_method = class_shift_method
        self.fingerprint_feature = fingerprint_feature
        self.polynomial_features = polynomial_features
        self.subsample_samples = subsample_samples

        self.inference_config = ModelInterfaceConfig(
            FEATURE_SHIFT_METHOD=self.feature_shift_method,
            CLASS_SHIFT_METHOD=self.class_shift_method,
            FINGERPRINT_FEATURE=self.fingerprint_feature,
            POLYNOMIAL_FEATURES=self.polynomial_features,
            SUBSAMPLE_SAMPLES=self.subsample_samples,
        )

        super().__init__(
            # Base training parameters
            epochs=self.n_estimators,
            learning_rate=None,
            batch_size=None,
            # Base early stopping parameters
            use_early_stopping=None,
            early_stopping_rounds=None,
            # Base preprocessing parameters
            feature_scaling=feature_scaling,
            standardize_targets=standardize_targets,
            clip_features=clip_features,
            clip_outputs=clip_outputs,
            # Base dimensionality reduction parameters
            use_pca=use_pca,
            n_pca_components=n_pca_components,
            # Base system and utility parameters
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

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
        self.best_iteration = self.epochs
