from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler


class BaseTabularRegressor(BaseEstimator, RegressorMixin, ABC):
    """
    Base class for tabular regressors with unified interface.

    Provides:
    - Unified parameter interface
    - Common preprocessing (scaling, standardization)
    - Early stopping bookkeeping
    - Common fit/predict patterns

    Subclasses should implement:
    - _create_model(): Create the underlying model
    - _fit_model(): Model-specific fitting logic
    - _predict_model(): Model-specific prediction logic
    """

    # Unified parameter mapping
    PARAM_MAPPING = {
        "epochs": [
            "epochs",
            "n_epochs",
            "n_estimators",
        ],
        "learning_rate": [
            "learning_rate",
            "lr",
        ],
        "early_stopping_rounds": [
            "early_stopping_rounds",
            "early_stopping_patience",
            "early_stopping_additive_patience",
        ],
        "use_early_stopping": [
            "use_early_stopping",
        ],
        "random_state": [
            "random_state",
            "seed",
        ],
        "verbose": [
            "verbose",
            "verbosity",
        ],
    }

    def __init__(
        self,
        # Common parameters with unified names
        epochs: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: Optional[int] = 16,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 40,
        # Misc
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        # Preprocessing
        feature_scaling: Optional[str] = None,  # 'standard', 'robust', or None
        standardize_targets: bool = False,
        clip_features: bool = False,
        clip_outputs: bool = False,
        **kwargs,
    ):
        """
        Initialize base regressor with common parameters.

        Subclasses can accept additional parameters in kwargs.
        """
        # Common parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds

        # Preprocessing
        self.feature_scaling = feature_scaling
        self.standardize_targets = standardize_targets
        self.clip_features = clip_features
        self.clip_outputs = clip_outputs

        # Misc
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Store extra kwargs for subclasses
        self._extra_params = kwargs

        # Will be set during fit
        self.best_iteration = None
        self._feature_scaler = None
        self._target_stats = None

    def _apply_feature_preprocessing(self, X, fit=False):
        """Apply feature preprocessing (scaling, clipping)."""
        X = np.asarray(X)

        if self.feature_scaling:
            if fit:
                if self.feature_scaling == "standard":
                    self._feature_scaler = StandardScaler()
                elif self.feature_scaling == "robust":
                    self._feature_scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.feature_scaling}")
                X = self._feature_scaler.fit_transform(X)
            elif self._feature_scaler is not None:
                X = self._feature_scaler.transform(X)

        if self.clip_features:
            X = np.clip(X, -3, 3)

        return X

    def _apply_target_preprocessing(self, y, fit=False):
        """Apply target preprocessing (standardization)."""
        y = np.asarray(y).flatten()

        if fit:
            self._target_stats = {
                "min": np.min(y),
                "max": np.max(y),
                "mean": np.mean(y),
                "std": np.std(y) if np.std(y) > 0 else 1.0,
            }

        if self.standardize_targets and self._target_stats:
            y = (y - self._target_stats["mean"]) / self._target_stats["std"]

        return y

    def _inverse_target_transform(self, y):
        """Inverse target preprocessing."""
        if self.standardize_targets and self._target_stats:
            y = y * self._target_stats["std"] + self._target_stats["mean"]

        if self.clip_outputs and self._target_stats:
            y = np.clip(y, self._target_stats["min"], self._target_stats["max"])

        return y

    def _prepare_eval_set(self, eval_set):
        """Prepare validation set with preprocessing."""
        if eval_set is None or len(eval_set) == 0:
            return None

        X_val, y_val = eval_set[0]
        X_val = self._apply_feature_preprocessing(X_val, fit=False)
        y_val = self._apply_target_preprocessing(y_val, fit=False)
        return [(X_val, y_val)]

    def _get_model_params(self):
        """
        Get parameters mapped to model-specific names.
        Subclasses can override to add custom mappings.
        """
        params = {}
        expected_params = self._get_expected_params()

        # First, collect all parameters from _extra_params that are expected
        for param_name in expected_params:
            if param_name in self._extra_params:
                params[param_name] = self._extra_params[param_name]

        # Then apply parameter mapping for unified names
        for unified_name, possible_names in self.PARAM_MAPPING.items():
            if hasattr(self, unified_name):
                value = getattr(self, unified_name)
                # Find which name the model expects
                for name in possible_names:
                    if name in expected_params:
                        params[name] = value
                        break

        # Also check if any expected params exist as attributes on self
        # (for backwards compatibility and flexibility)
        for param_name in expected_params:
            if param_name not in params and hasattr(self, param_name):
                params[param_name] = getattr(self, param_name)

        return params

    @abstractmethod
    def _get_expected_params(self):
        """Return list of parameter names expected by the underlying model."""
        pass

    @abstractmethod
    def _create_model(self, n_features):
        """Create the underlying model instance."""
        pass

    @abstractmethod
    def _fit_model(self, X, y, eval_set=None):
        """
        Fit the underlying model.
        Should set self.best_iteration if using early stopping.
        """
        pass

    def _predict_model(self, X):
        """
        Make predictions with the underlying model.
        Default implementation calls self.model.predict(X).
        """
        return self.model.predict(X)

    def fit(self, X, y, eval_set=None, verbose=False):
        """Common fit interface."""
        # Apply preprocessing
        X = self._apply_feature_preprocessing(X, fit=True)
        y = self._apply_target_preprocessing(y, fit=True)
        eval_set = self._prepare_eval_set(eval_set)

        # Create model
        n_features = X.shape[1]
        self.model = self._create_model(n_features)

        # Fit model (subclass-specific)
        self._fit_model(X, y, eval_set)

        return self

    def predict(self, X):
        """Common predict interface."""
        X = self._apply_feature_preprocessing(X, fit=False)
        y_pred = self._predict_model(X)
        y_pred = self._inverse_target_transform(y_pred)
        return y_pred
