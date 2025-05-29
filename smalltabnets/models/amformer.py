from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from ..modules.amformer.AM_Former import AMFormer
from .base import BaseTabularRegressor


class AMFormerRegressor(BaseTabularRegressor):
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
        # AM-Former specific parameters
        dim: int = 32,
        depth: int = 2,
        heads: int = 2,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        use_cls_token: bool = False,
        groups: int = 1,
        sum_num_per_group: int = 1,
        prod_num_per_group: int = 1,
        cluster: bool = True,
        target_mode: Optional[str] = None,
        token_descent: bool = False,
        use_prod: bool = True,
        num_special_tokens: int = 2,
        qk_relu: bool = False,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.use_cls_token = use_cls_token
        self.groups = groups
        self.sum_num_per_group = sum_num_per_group
        self.prod_num_per_group = prod_num_per_group
        self.cluster = cluster
        self.target_mode = target_mode
        self.token_descent = token_descent
        self.use_prod = use_prod
        self.num_special_tokens = num_special_tokens
        self.qk_relu = qk_relu

        self._embedder = None

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
        # The LimeSoDa datasets used in the benchmark contain only numerical
        # columns, so we initialize with num_cate = 0 and categories = []

        def _to_list(value, depth: int):
            # Harmonise list-valued parameters
            return value if isinstance(value, (list, tuple)) else [value] * depth

        groups_list = _to_list(self.groups, self.depth)
        sum_list = _to_list(self.sum_num_per_group, self.depth)
        prod_list = _to_list(self.prod_num_per_group, self.depth)

        args = SimpleNamespace(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            use_cls_token=self.use_cls_token,
            groups=groups_list,
            sum_num_per_group=sum_list,
            prod_num_per_group=prod_list,
            cluster=self.cluster,
            target_mode=self.target_mode,
            num_cont=n_features,
            num_cate=0,
            token_descent=self.token_descent,
            use_prod=self.use_prod,
            num_special_tokens=self.num_special_tokens,
            categories=[],  # no categorical inputs
            out=1,  # regression
            use_sigmoid=False,
            qk_relu=self.qk_relu,
        )

        model = AMFormer(args)
        return model.to(self.device)

    def _forward_pass(self, batch):
        dummy_label = torch.zeros(
            batch["inputs"].shape[0],
            device=self.device,
        )
        out, _ = self.model(
            x_categ=None,
            x_numer=batch["inputs"],
            label=dummy_label,
        )
        return out.squeeze(-1)

    def _fit_model(self, X, y, eval_set=None):
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
