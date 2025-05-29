from typing import List, Optional

import torch
import torch.nn as nn

from ..modules.ft_transformer.ft_transformer import FTTransformer
from .base import BaseTabularRegressor


class FTTransformerRegressor(BaseTabularRegressor):
    def __init__(
        self,
        *,
        # FT-Transformer specific parameters
        token_bias: bool = True,
        n_layers: int = 4,
        d_token: int = 128,
        n_heads: int = 8,
        d_ffn_factor: float = 2.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        activation: str = "geglu",
        prenormalization: bool = True,
        initialization: str = "kaiming",
        kv_compression: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        # Accept all base parameters via **kwargs
        **kwargs
    ):
        # Store model-specific parameters
        self.token_bias = token_bias
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_ffn_factor = d_ffn_factor
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.activation = activation
        self.prenormalization = prenormalization
        self.initialization = initialization
        self.kv_compression = kv_compression
        self.kv_compression_sharing = kv_compression_sharing

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features: int) -> nn.Module:
        model = FTTransformer(
            # Tokenizer
            d_numerical=n_features,
            categories=None,
            token_bias=self.token_bias,
            # Transformer
            n_layers=self.n_layers,
            d_token=self.d_token,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            activation=self.activation,
            prenormalization=self.prenormalization,
            initialization=self.initialization,
            # Linformer style compression
            kv_compression=self.kv_compression,
            kv_compression_sharing=self.kv_compression_sharing,
            # Output
            d_out=1,  # regression
        )
        return model.to(self.device)

    def _fit_model(self, X, y, eval_set=None):
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
