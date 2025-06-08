from typing import Any, Dict, Optional

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn

from ..modules.modernnca.modernnca import ModernNCA
from .base import BaseTabularRegressor


class ModernNCARegressor(BaseTabularRegressor):
    def __init__(
        self,
        *,
        # Modern-NCA specific parameters
        dim: int = 128,
        d_block: int = 256,
        n_blocks: int = 0,
        dropout: float = 0.0,
        temperature: float = 1.0,
        sample_rate: float = 0.3,
        # Embeddings
        use_embeddings: bool = False,
        embedding_type: str = "piecewise_linear",  # "piecewise_linear" or "linear"
        embedding_dim: int = 16,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):

        self.dim = dim
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.temperature = temperature
        self.sample_rate = sample_rate

        # Embeddings
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_embeddings = None
        self.bins = None

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features: int):
        model = ModernNCA(
            d_in=n_features,
            d_num=n_features,
            d_out=1,  # regression
            dim=self.dim,
            dropout=self.dropout,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            num_embeddings=self.num_embeddings,
            bins=self.bins,
            temperature=self.temperature,
            sample_rate=self.sample_rate,
        )
        return model.to(self.device)

    def _prepare_embeddings(self, X):
        """
        Prepare numerical embeddings configuration.
        Subclasses can override for custom embedding logic.
        """
        if not self.use_embeddings:
            return

        if self.embedding_type == "piecewise_linear":
            # In the original TabM implementation n_bins is fixed to 48
            # However it should always be less than len(X) - 1, which is a problem for our small datasets
            # We also want to avoid too many bins, so we use a simple heuristic
            # We set a maximum of 48 bins, and a minimum of len(X) // 2 bins.
            n_bins = min(48, len(X) // 2)
            # Compute bins using quantiles
            self.bins = rtdl_num_embeddings.compute_bins(
                torch.as_tensor(X),
                n_bins=n_bins,
            )
            self.num_embeddings = {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": self.embedding_dim,
                "activation": False,
                "version": "B",
            }
        elif self.embedding_type == "linear":
            self.num_embeddings = {
                "type": "LinearEmbeddings",
                "d_embedding": self.embedding_dim,
            }
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def _prepare_batch(
        self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Prepare a batch for training.
        Subclasses can override to add special handling.

        Returns dict with at least 'inputs' and 'targets' keys.
        Additional keys can be added for models with special requirements.
        """
        # Additional check for ModernNCA
        # the batch should contain at least 2 samples
        # so we return an empty batch if indices are too small
        # and let the training loop handle it
        if len(indices) <= 1:
            empty_shape = (0, *X_tensor.shape[1:])
            return {
                "inputs": torch.empty(empty_shape, device=self.device),
                "targets": torch.empty(empty_shape, device=self.device),
            }

        return {"inputs": X_tensor[indices], "targets": y_tensor[indices]}

    def fit(self, X, y, eval_set=None, verbose=False):
        """Common fit interface overriding with embeddings init logic."""
        # Apply preprocessing
        X = self._apply_feature_preprocessing(X, fit=True)
        if self.use_embeddings:
            self._prepare_embeddings(X)

        y = self._apply_target_preprocessing(y, fit=True)
        eval_set = self._prepare_eval_set(eval_set)

        # Create model
        n_features = X.shape[1]
        self.model = self._create_model(n_features)

        # Fit model (subclass-specific)
        self._fit_model(X, y, eval_set)

        return self

    def _fit_model(self, X, y, eval_set=None):
        # Store retrieval memory for later prediction
        self._candidate_x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self._candidate_y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return self._fit_pytorch_model(X, y, eval_set)

    def _forward_pass(self, batch):
        xb = batch["inputs"]
        yb = batch["targets"]
        output = self.model(
            xb,
            yb,
            candidate_x=self._candidate_x,
            candidate_y=self._candidate_y,
            is_train=True,
        )
        return output

    def _predict_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        """ModernNCA-specific prediction with retrieval memory."""
        if not hasattr(self, "_candidate_x"):
            raise RuntimeError("Model must be fitted before calling predict.")

        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), 8192):
                batch = X_tensor[start : start + 8192]
                out = self.model(
                    batch,
                    None,  # No targets during inference
                    candidate_x=self._candidate_x,
                    candidate_y=self._candidate_y,
                    is_train=False,  # Inference mode
                )
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
