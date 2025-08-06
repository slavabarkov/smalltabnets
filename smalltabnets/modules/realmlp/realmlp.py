import math
from typing import Sequence

import rtdl_num_embeddings
import torch
import torch.nn as nn


class ScalingLayer(nn.Module):
    """Element-wise learnable scaling"""

    def __init__(self, n_features: int, init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((n_features,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LinearNTP(nn.Linear):
    """Linear layer with NT parametrisation and he+5 bias initialisation"""

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0)  # unit-variance
        if self.bias is not None:
            # he+5 bias initialisation, trim negative bias
            nn.init.normal_(self.bias, mean=0.0, std=1.0)
            self.bias.data = self.bias.data.clamp_min(0.0) - 5.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.in_features)
        return (x @ self.weight.T) * scale + self.bias


class ParametricMish(nn.Module):
    """Learnable mixture of identity and Mish"""

    def __init__(self, n_features: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n_features,), init_alpha))

    @staticmethod
    def _mish(x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * (self._mish(x) - x)


class RealMLPRegressor(nn.Module):
    """
    Implementation of RealMLP-TD for Regression with numerical features
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (256, 256, 256),
        dropout: float = 0.15,
        use_front_scale: bool = True,
        use_parametric_act: bool = True,
        use_num_emb: bool = True,
        num_emb_type: str = "piecewise_linear",
        num_emb_dim: int = 16,
        num_emb_bins=None,
        out_features: int = 1,
    ):
        super().__init__()

        # Numerical embeddings
        if use_num_emb:
            if num_emb_type == "piecewise_linear":
                if num_emb_bins is None:
                    raise ValueError("Bins must be provided for PL embeddings.")
                self.embedding = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                    bins=num_emb_bins,
                    d_embedding=num_emb_dim,
                    activation=False,
                    version="B",
                )
            else:
                raise NotImplementedError(
                    f"Numerical embedding type '{num_emb_type}' is not implemented."
                )

            in_linear = in_features * num_emb_dim
        else:
            self.embedding = nn.Identity()
            in_linear = in_features

        layers = []

        # Optional scaling layer
        if use_front_scale:
            layers.append(ScalingLayer(in_linear, init=1.0))

        # Hidden blocks
        n_in = in_linear
        for _, n_out in enumerate(hidden_sizes):
            layers.append(LinearNTP(n_in, n_out, bias=True))
            if use_parametric_act:
                layers.append(ParametricMish(n_out))
            else:
                layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout))
            n_in = n_out

        # Last linear layer
        layers.append(LinearNTP(n_in, out_features, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.flatten(1, -1)
        return self.net(x)
