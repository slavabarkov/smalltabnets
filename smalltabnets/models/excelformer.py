from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..modules.excelformer.excelformer import ExcelFormer
from .base import BaseTabularRegressor


class ExcelFormerRegressor(BaseTabularRegressor):
    def __init__(
        self,
        *,
        # ExcelFormer specific parameters
        token_bias: bool = True,
        n_layers: int = 2,
        d_token: int = 32,
        n_heads: int = 2,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        prenormalization: bool = False,
        kv_compression: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        init_scale: float = 0.1,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        # Store ExcelFormer-specific hyper-parameters
        self.token_bias = token_bias
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.kv_compression = kv_compression
        self.kv_compression_sharing = kv_compression_sharing
        self.init_scale = init_scale

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features: int) -> nn.Module:
        model = ExcelFormer(
            d_numerical=n_features,
            categories=None,  # all-numerical data set
            token_bias=self.token_bias,
            n_layers=self.n_layers,
            d_token=self.d_token,
            n_heads=self.n_heads,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            prenormalization=self.prenormalization,
            kv_compression=self.kv_compression,
            kv_compression_sharing=self.kv_compression_sharing,
            d_out=1,  # regression
            init_scale=self.init_scale,
        )
        return model.to(self.device)

    def _fit_model(self, X, y, eval_set=None):
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
