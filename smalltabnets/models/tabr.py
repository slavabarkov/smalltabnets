import math
from typing import Optional

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn

from ..modules.tabr.tabr import TabR
from .base import BaseTabularRegressor


class TabRRegressor(BaseTabularRegressor):
    def __init__(
        self,
        # Base (shared) parameters
        epochs: int = 256,
        learning_rate: float = 2e-3,
        batch_size: int = 32,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        # Dimensionality reduction
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # TabR specific
        d_main: int = 64,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 2,
        predictor_n_blocks: int = 2,
        mixer_normalization: str = "auto",
        context_dropout: float = 0.1,
        dropout0: float = 0.1,
        dropout1: str = "dropout0",  # may be "dropout0" as in paper
        normalization: str = "LayerNorm",
        activation: str = "ReLU",
        context_size: int = 16,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
        # Embeddings
        use_embeddings: bool = True,
        embedding_type: str = "piecewise_linear",  # or "linear", "lr", "plr", etc.
        embedding_dim: int = 16,
        # Misc
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        **kwargs,
    ):
        # Store TabR specific parameters
        self.d_main = d_main
        self.d_multiplier = d_multiplier
        self.encoder_n_blocks = encoder_n_blocks
        self.predictor_n_blocks = predictor_n_blocks
        self.mixer_normalization = mixer_normalization
        self.context_dropout = context_dropout
        self.dropout0 = dropout0
        self.dropout1 = dropout1
        self.normalization = normalization
        self.activation = activation
        self.context_size = context_size
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size

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
            "d_main",
            "d_multiplier",
            "encoder_n_blocks",
            "predictor_n_blocks",
            "mixer_normalization",
            "context_dropout",
            "dropout0",
            "dropout1",
            "normalization",
            "activation",
            "context_size",
            "memory_efficient",
            "candidate_encoding_batch_size",
            "device",
            "random_state",
            "verbose",
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
        ]

    def _create_model(self, n_features: int):
        """
        Instantiate TabR.  All features in LimeSoDa are treated as numerical
        (n_num_features = n_features, n_cat_features = 0).
        """
        model = TabR(
            # data-related
            n_num_features=n_features,
            n_cat_features=0,
            n_classes=1,  # regression
            num_embeddings=self.num_embeddings,
            bins=self.bins,
            # architecture / training params
            d_main=self.d_main,
            d_multiplier=self.d_multiplier,
            encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks,
            mixer_normalization=self.mixer_normalization,
            context_dropout=self.context_dropout,
            dropout0=self.dropout0,
            dropout1=self.dropout1,
            normalization=self.normalization,
            activation=self.activation,
            memory_efficient=self.memory_efficient,
            candidate_encoding_batch_size=self.candidate_encoding_batch_size,
        )
        return model.to(self.device)

    def _prepare_batch(self, X_tensor, y_tensor, indices):
        """TabR needs to prepare candidate memory for each batch."""
        batch = super()._prepare_batch(X_tensor, y_tensor, indices)

        # Build candidate memory WITHOUT current batch
        n_train = len(X_tensor)
        mask = torch.ones(n_train, dtype=torch.bool, device=self.device)
        mask[indices] = False

        batch["candidate_x_num"] = X_tensor[mask]
        batch["candidate_y"] = y_tensor[mask]
        batch["candidate_x_cat"] = None
        batch["context_size"] = self.context_size

        return batch

    def _forward_pass(self, batch):
        """TabR has special forward signature with candidates."""
        return self.model(
            x_num=batch["inputs"],
            x_cat=None,
            y=batch["targets"],
            candidate_x_num=batch["candidate_x_num"],
            candidate_x_cat=batch["candidate_x_cat"],
            candidate_y=batch["candidate_y"],
            context_size=batch["context_size"],
            is_train=True,
        ).squeeze(-1)

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
        """Store candidates for inference, then train."""
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # Store for inference
        self._candidate_x_num = X_tensor
        self._candidate_y = y_tensor
        self._candidate_x_cat = None

        # Use base training
        self._fit_pytorch_model(X, y, eval_set)

    # Custom predict_tensor for inference with candidates
    def _predict_tensor(self, X_tensor):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], 8192):
                xb = X_tensor[start : start + 8192]
                out = self.model(
                    x_num=xb,
                    x_cat=None,
                    y=None,
                    candidate_x_num=self._candidate_x_num,
                    candidate_x_cat=self._candidate_x_cat,
                    candidate_y=self._candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
