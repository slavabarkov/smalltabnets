from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


class BaseTabularRegressor(BaseEstimator, RegressorMixin, ABC):
    """
    Base class for tabular regressors with unified interface.

    Provides:
    - Unified parameter interface
    - Common preprocessing (scaling, standardization, dimensionality reduction)
    - Early stopping bookkeeping
    - Common fit/predict patterns
    - Common PyTorch training loop for models that use it

    Subclasses should implement:
    - _create_model(): Create the underlying model
    - _fit_model(): Model-specific fitting logic (or use _fit_pytorch_model)
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
        # Dimensionality reduction
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
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

        # Dimensionality reduction
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self._pca = None

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
        self._original_n_features = None

    def _apply_feature_preprocessing(self, X, fit=False):
        """Apply feature preprocessing (scaling, clipping, PCA)."""
        X = np.asarray(X)

        if fit:
            self._original_n_features = X.shape[1]

        # Apply scaling
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

        # Apply clipping
        if self.clip_features:
            X = np.clip(X, -3, 3)

        # Apply dimensionality reduction
        if self.use_pca:
            if fit:
                n_components = self.n_pca_components
                if n_components is None:
                    # Default to keeping 95% of variance
                    n_components = 0.95
                elif n_components > X.shape[1]:
                    n_components = X.shape[1]

                self._pca = PCA(
                    n_components=n_components, random_state=self.random_state
                )
                X = self._pca.fit_transform(X)

                if self.verbose:
                    actual_components = self._pca.n_components_
                    explained_var = self._pca.explained_variance_ratio_.sum()
                    print(
                        f"PCA: {self._original_n_features} -> {actual_components} features "
                        f"({explained_var:.1%} variance explained)"
                    )
            elif self._pca is not None:
                X = self._pca.transform(X)

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

    # ========================================================================
    # Common PyTorch training logic
    # ========================================================================

    def _get_optimizer(self, parameters) -> torch.optim.Optimizer:
        """
        Create optimizer for training.
        Subclasses can override to use different optimizers.
        """
        return torch.optim.Adam(parameters, lr=self.learning_rate)

    def _get_criterion(self):
        """
        Get loss criterion.
        Subclasses can override for different loss functions.
        """
        return nn.MSELoss()

    def _prepare_batch(
        self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Prepare a batch for training.
        Subclasses can override to add special handling.

        Returns dict with at least 'inputs' and 'targets' keys.
        Additional keys can be added for models with special requirements.
        """
        return {"inputs": X_tensor[indices], "targets": y_tensor[indices]}

    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform forward pass.
        Subclasses can override for models with special forward requirements.

        Default assumes model takes a single tensor input.
        """
        return self.model(batch["inputs"], None).squeeze(-1)

    def _compute_loss(
        self, predictions: torch.Tensor, batch: Dict[str, Any], criterion
    ) -> torch.Tensor:
        """
        Compute loss.
        Subclasses can override for special loss calculations.
        """
        targets = batch["targets"]
        if predictions.ndim == 2 and targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        return criterion(predictions, targets)

    def _optimizer_step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor):
        """
        Perform optimizer step.
        Subclasses can override to add gradient clipping, etc.
        """
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    @torch.inference_mode()
    def _evaluate_pytorch_model(
        self, X_val: torch.Tensor, y_val: torch.Tensor, criterion
    ) -> float:
        """
        Evaluate model on validation set.
        Returns validation RMSE by default.
        """
        self.model.eval()
        val_preds = self._predict_tensor(X_val)
        val_loss = criterion(torch.as_tensor(val_preds, device=self.device), y_val)
        return float(torch.sqrt(val_loss).item())

    def _predict_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        """
        Make predictions on a tensor.
        Subclasses should override if they have special prediction logic.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), 8192):
                batch = {"inputs": X_tensor[start : start + 8192]}
                out = self._forward_pass(batch)
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _fit_pytorch_model(self, X, y, eval_set=None):
        """
        Common PyTorch training loop.
        Subclasses can use this directly or override _fit_model with custom logic.
        """
        # Convert to tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # Set random seeds
        # torch.manual_seed(self.random_state) #!!!
        rng = np.random.RandomState(self.random_state)

        # Create optimizer and criterion
        optimizer = self._get_optimizer(self.model.parameters())
        criterion = self._get_criterion()

        # Validation setup
        X_val = y_val = None
        if self.use_early_stopping and eval_set and len(eval_set) > 0:
            X_val_np, y_val_np = eval_set[0]
            X_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=self.device)

        # Early stopping variables
        best_metric = float("inf")
        best_state = None
        patience_left = self.early_stopping_rounds if X_val is not None else None
        self.best_iteration = self.epochs - 1  # Default to last epoch

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()

            # Shuffle data
            perm = rng.permutation(len(X_tensor))

            # Mini-batch training
            for start in range(0, len(perm), self.batch_size):
                indices = perm[start : start + self.batch_size]

                # Prepare batch
                batch = self._prepare_batch(X_tensor, y_tensor, indices)

                # Forward pass
                predictions = self._forward_pass(batch)

                # Compute loss
                loss = self._compute_loss(predictions, batch, criterion)

                # Optimizer step
                self._optimizer_step(optimizer, loss)

            # Validation and early stopping
            if X_val is not None:
                val_metric = self._evaluate_pytorch_model(X_val, y_val, criterion)

                if val_metric < best_metric:
                    best_metric = val_metric
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self.best_iteration = epoch
                    if patience_left is not None:
                        patience_left = self.early_stopping_rounds
                else:
                    if patience_left is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            if self.verbose:
                                print(f"Early stopping triggered at epoch {epoch}")
                            break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ========================================================================
    # Abstract methods
    # ========================================================================

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

        PyTorch models can simply call self._fit_pytorch_model(X, y, eval_set)
        if they follow the standard training pattern.
        """
        pass

    def _predict_model(self, X):
        """
        Make predictions with the underlying model.
        Default implementation calls self.model.predict(X).
        PyTorch models will typically override this.
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
