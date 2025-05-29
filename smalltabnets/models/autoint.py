from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..modules.autoint.autoint import AutoInt
from .base import BaseTabularRegressor


class AutoIntRegressor(BaseTabularRegressor):
    def __init__(
        self,
        # Base training parameters
        epochs: int = 256,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        # Base early stopping parameters
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        # Base preprocessing parameters
        feature_scaling: bool = "robust",
        standardize_targets: bool = True,
        clip_features: bool = False,
        clip_outputs: bool = False,
        # Base dimensionality reduction parameters
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # Base system and utility parameters
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        # AutoInt specific parameters
        n_layers: int = 2,
        d_token: int = 32,
        n_heads: int = 2,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        activation: str = "relu",  # must stay "relu" (see assertion)
        prenormalization: bool = False,  # must stay False  (see assertion)
        initialization: str = "kaiming",  # "kaiming" or "xavier"
        kv_compression: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        **kwargs,
    ):
        # Store AutoInt-specific hyper-parameters
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.activation = activation
        self.prenormalization = prenormalization
        self.initialization = initialization
        self.kv_compression = kv_compression
        self.kv_compression_sharing = kv_compression_sharing

        super().__init__(
            # Base training parameters
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            # Base early stopping parameters
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
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

    def _create_model(self, n_features: int) -> nn.Module:
        model = AutoInt(
            d_numerical=n_features,
            categories=None,  # all-numerical data set
            n_layers=self.n_layers,
            d_token=self.d_token,
            n_heads=self.n_heads,
            attention_dropout=self.attention_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            prenormalization=self.prenormalization,
            initialization=self.initialization,
            kv_compression=self.kv_compression,
            kv_compression_sharing=self.kv_compression_sharing,
            d_out=1,  # regression
        )
        return model.to(self.device)

    def _fit_model(self, X, y, eval_set=None):
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
