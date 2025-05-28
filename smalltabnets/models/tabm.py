import math
from typing import Optional

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn

from ..modules.tabm import TabMModel, make_parameter_groups
from .base import BaseTabularRegressor


class TabMRegressor(BaseTabularRegressor):
    """TabM regressor with unified interface."""

    def __init__(
        self,
        # Base parameters
        epochs=1000,
        learning_rate=2e-3,
        batch_size=16,
        use_early_stopping=True,
        early_stopping_rounds=16,
        # Architecture parameters
        arch_type: str = "tabm",
        k: int = 32,
        # Parameters for building the backbone
        n_blocks=3,
        d_block=512,
        dropout=0.1,
        activation="ReLU",
        # Training parameters
        weight_decay=3e-4,
        beta1=0.9,
        beta2=0.999,
        share_training_batches: bool = True,
        gradient_clipping_norm: Optional[float] = 1.0,
        # Embeddings
        use_embeddings: bool = True,
        embedding_type: str = "piecewise_linear",  # or "linear", "lr", "plr", etc.
        embedding_dim: int = 16,
        # Dimensionality reduction
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # Misc
        device=None,
        random_state=42,
        verbose=0,
        **kwargs,
    ):
        self.arch_type = arch_type
        self.k = k

        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.activation = activation

        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_clipping_norm = gradient_clipping_norm
        self.share_training_batches = share_training_batches

        # Embeddings
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_embeddings = None
        self.bins = None

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
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            "k",
            "arch_type",
            "n_blocks",
            "d_block",
            "dropout",
            "activation",
            "weight_decay",
            "beta1",
            "beta2",
            "gradient_clipping_norm",
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
            "device",
            "random_state",
            "verbose",
            "use_embeddings",
            "embedding_type",
            "embedding_dim",
            "use_pca",
            "n_pca_components",
        ]

    def _prepare_embeddings(self, X):
        """
        Prepare numerical embeddings configuration.
        Subclasses can override for custom embedding logic.
        """
        if not self.use_embeddings:
            return

        if self.embedding_type == "piecewise_linear":
            self.bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(X))
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

    def _create_model(self, n_features):
        model = TabMModel(
            n_num_features=n_features,
            cat_cardinalities=[],
            n_classes=None,  # regression
            backbone={
                "type": "MLP",
                "n_blocks": self.n_blocks,
                "d_block": self.d_block,
                "dropout": self.dropout,
                "activation": self.activation,
            },
            bins=self.bins,
            num_embeddings=self.num_embeddings,
            arch_type=self.arch_type,
            k=self.k,
            share_training_batches=self.share_training_batches,
        )

        return model.to(self.device)

    def _get_optimizer(self, parameters):
        """TabM uses AdamW with custom parameter groups."""

        return torch.optim.AdamW(
            make_parameter_groups(self.model),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

    def _optimizer_step(self, optimizer, loss):
        """Add gradient clipping."""
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.gradient_clipping_norm:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping_norm
            )

        optimizer.step()

    def _compute_loss(self, predictions, batch, criterion):
        """Special loss for ensemble."""
        if self.k:  # If using ensemble
            targets = batch["targets"].repeat_interleave(self.k)
            return criterion(predictions.flatten(), targets)
        else:
            return super()._compute_loss(predictions, batch, criterion)

    def _forward_pass(self, batch):
        """TabM returns ensemble predictions."""
        out = self.model(batch["inputs"])
        return out.squeeze(-1)  # Shape: (batch, k) for ensemble

    def _predict_tensor(self, X_tensor):
        """Average ensemble predictions."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), 8192):
                batch = X_tensor[start : start + 8192]
                out = self.model(batch).squeeze(-1).mean(1)  # Average over k
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

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
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
