from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..modules.autoint.autoint import AutoInt
from .base import BaseTabularRegressor


class AutoIntRegressor(BaseTabularRegressor):
    def __init__(
        self,
        *,
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
        # Accept all base parameters via **kwargs
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

        # Pass all base parameters to parent
        super().__init__(**kwargs)

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
