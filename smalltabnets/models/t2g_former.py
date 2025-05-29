from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..modules.t2g_former.t2g_former import T2GFormer
from .base import BaseTabularRegressor


class T2GFormerRegressor(BaseTabularRegressor):
    def __init__(
        self,
        *,
        # T2G Former specific parameters
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
        sym_weight: bool = True,
        sym_topology: bool = False,
        nsi: bool = True,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
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
        self.sym_weight = sym_weight
        self.sym_topology = sym_topology
        self.nsi = nsi

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features: int) -> nn.Module:
        model = T2GFormer(
            #  tokenizer
            d_numerical=n_features,
            categories=None,  # purely numerical
            token_bias=self.token_bias,
            # transformer
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
            # Graph-estimator flags
            sym_weight=self.sym_weight,
            sym_topology=self.sym_topology,
            nsi=self.nsi,
            d_out=1,  # regression
        )
        return model.to(self.device)

    def _fit_model(self, X, y, eval_set=None):
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
