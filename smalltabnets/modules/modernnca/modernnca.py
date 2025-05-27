# Code from https://github.com/LAMDA-Tabular/TALENT which is licensed MIT

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LREmbeddings(nn.Sequential):
    """The LR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'."""  # noqa: E501

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__(LinearEmbeddings(n_features, d_embedding), nn.ReLU())


class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )


class PBLDEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
        plr_act_name: str = "relu",
        plr_use_densenet: bool = True,
    ):
        super().__init__()
        print(f"Constructing PBLD embeddings")
        hidden_2 = d_embedding - 1 if plr_use_densenet else d_embedding
        self.weight_1 = nn.Parameter(
            frequency_scale * torch.randn(n_features, 1, n_frequencies)
        )
        self.weight_2 = nn.Parameter(
            (-1 + 2 * torch.rand(n_features, n_frequencies, hidden_2))
            / np.sqrt(n_frequencies)
        )
        self.bias_1 = nn.Parameter(
            np.pi * (-1 + 2 * torch.rand(n_features, 1, n_frequencies))
        )
        self.bias_2 = nn.Parameter(
            (-1 + 2 * torch.rand(n_features, 1, hidden_2)) / np.sqrt(n_frequencies)
        )
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = x + self.bias_1
        # x = torch.sin(x)
        x = torch.cos(x)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == "relu":
            x = torch.relu(x)
        elif self.plr_act_name == "linear":
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: Union[None, int] = None,
        d_out: Union[None, int] = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "SELU",
    ) -> None:
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        LinearEmbeddings,
        LREmbeddings,
        PLREmbeddings,
        MLP,
        PBLDEmbeddings,
    ]
}


def make_module(spec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, str):
        Module = getattr(nn, spec, None)
        if Module is None:
            Module = _CUSTOM_MODULES[spec]
        else:
            assert spec not in _CUSTOM_MODULES
        return make_module(Module, *args, **kwargs)
    elif isinstance(spec, dict):
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop("type"), *args, **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()


class MLP_Block(nn.Module):
    def __init__(self, d_in: int, d: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num: int,
        d_out: int,
        dim: int,
        dropout: int,
        d_block: int,
        n_blocks: int,
        num_embeddings: Optional[dict],
        temperature: float = 1.0,
        sample_rate: float = 0.8,
    ) -> None:

        super().__init__()
        self.d_in = (
            d_in
            if num_embeddings is None
            else d_num * num_embeddings["d_embedding"] + d_in - d_num
        )
        self.d_out = d_out
        self.d_num = d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T = temperature
        self.sample_rate = sample_rate
        if n_blocks > 0:
            self.post_encoder = nn.Sequential(
                *[MLP_Block(dim, d_block, dropout) for _ in range(n_blocks)],
                nn.BatchNorm1d(dim),
            )
        self.encoder = nn.Linear(self.d_in, dim)
        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block = MLP_Block(self.dim, self.d_block, self.dropout)
        return block

    def forward(
        self,
        x,
        y,
        candidate_x,
        candidate_y,
        is_train,
    ):
        if is_train:
            data_size = candidate_x.shape[0]
            retrival_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size)[:retrival_size]
            candidate_x = candidate_x[sample_idx]
            candidate_y = candidate_y[sample_idx]
        if self.num_embeddings is not None and self.d_num > 0:
            x_num, x_cat = x[:, : self.d_num], x[:, self.d_num :]
            candidate_x_num, candidate_x_cat = (
                candidate_x[:, : self.d_num],
                candidate_x[:, self.d_num :],
            )
            x_num = self.num_embeddings(x_num).flatten(1)
            candidate_x_num = self.num_embeddings(candidate_x_num).flatten(1)
            x = torch.cat([x_num, x_cat], dim=-1)
            candidate_x = torch.cat([candidate_x_num, candidate_x_cat], dim=-1)
        # x=x.double()
        # candidate_x=candidate_x.double()
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)
        if self.n_blocks > 0:
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)
        if is_train:
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])

        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y = candidate_y.unsqueeze(-1)

        # calculate distance
        # default we use euclidean distance, however, cosine distance is also a good choice for classification.
        # Using cosine distance, you need to tune the temperature. You can add "temperature":["loguniform",1e-5,1] in the configs/opt_space/modernNCA.json file.
        distances = torch.cdist(x, candidate_x, p=2)
        # following is the code for cosine distance
        # x=F.normalize(x,p=2,dim=-1)
        # candidate_x=F.normalize(candidate_x,p=2,dim=-1)
        # distances=torch.mm(x,candidate_x.T)
        # distances=-distances
        distances = distances / self.T
        # remove the label of training index
        if is_train:
            distances = distances.fill_diagonal_(torch.inf)
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps = 1e-7
        if self.d_out > 1:
            # if task type is classification, since the logit is already normalized, we calculate the log of the logit
            # and use nll_loss to calculate the loss
            logits = torch.log(logits + eps)
        return logits.squeeze(-1)
