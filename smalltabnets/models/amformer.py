"""
Unified scikit-style wrapper for the AM-Former (FT-Transformer) network that
lives in `bodenboden.modules.amformer.AM_Former.FTTransformer`.

The LimeSoDa corpora used in the benchmark contain purely *numerical*
columns.  Consequently, AM-Former is instantiated with

    · num_cate = 0
    · categories = []

and all features are treated as continuous tokens.

The wrapper follows the same pattern as the other regressors in
`bodenboden/models/*` and inherits from `BaseTabularRegressor`, which
takes care of common pre-processing, early stopping bookkeeping and the
public `fit / predict` interface.
"""

from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..modules.amformer.AM_Former import FTTransformer
from .base import BaseTabularRegressor


class AMFormerRegressor(BaseTabularRegressor):
    """
    Thin adapter that exposes AM-Former through a familiar scikit-learn
    API and integrates seamlessly with `benchmark.py`.
    """

    # ------------------------------------------------------------------ #
    #  Construction / hyper-parameters
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        # >>> shared parameters (handled by the base-class) --------------
        epochs: int = 256,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        # >>> AM-Former architecture ------------------------------------
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
        # >>> absorb any extra YAML keys --------------------------------
        **kwargs,
    ):
        # Store architecture hyper-parameters for _create_model ----------
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

        # Let the common base-class deal with everything else ------------
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Base-class hooks
    # ------------------------------------------------------------------ #
    def _get_expected_params(self) -> List[str]:
        """
        Names that may appear in YAML / Optuna spaces and therefore need to
        be captured by `_get_model_params` (even though we build the network
        explicitly in `_create_model`).
        """
        return [
            # optimisation ------------------------------------------------
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            # architecture ------------------------------------------------
            "dim",
            "depth",
            "heads",
            "attn_dropout",
            "ff_dropout",
            "use_cls_token",
            "groups",
            "sum_num_per_group",
            "prod_num_per_group",
            "cluster",
            "target_mode",
            "token_descent",
            "use_prod",
            "num_special_tokens",
            "qk_relu",
            # misc --------------------------------------------------------
            "device",
            "random_state",
            "verbose",
            # preprocessing flags accepted in YAML ------------------------
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
        ]

    # ------------------------------------------------------------------ #
    def _create_model(self, n_features: int) -> nn.Module:
        """
        Build the underlying AM-Former (FT-Transformer) instance.

        LimeSoDa tables are *all numerical*, hence

            · num_cate = 0
            · categories = []
        """
        # Harmonise list-valued parameters ------------------------------
        def _to_list(value, depth: int):
            return value if isinstance(value, (list, tuple)) else [value] * depth

        groups_list = _to_list(self.groups, self.depth)
        sum_list = _to_list(self.sum_num_per_group, self.depth)
        prod_list = _to_list(self.prod_num_per_group, self.depth)

        # Collect everything into a simple namespace --------------------
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

        model = FTTransformer(args)
        return model.to(self.device)

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def _fit_model(self, X, y, eval_set=None):
        """
        Mini-batch optimisation with optional patience-based early stopping.
        """
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Validation data ------------------------------------------------
        if self.use_early_stopping and eval_set and len(eval_set) > 0:
            X_val_np, y_val_np = eval_set[0]
            X_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=self.device)
            patience_left = self.early_stopping_rounds
        else:
            X_val = y_val = None
            patience_left = None

        best_metric = float("inf")
        best_state: Optional[dict] = None
        self.best_iteration = self.epochs - 1  # will be updated if ES triggers

        # ----------------------------- optimisation loop -------------- #
        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(X_tensor))

            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb_num = X_tensor[idx]
                yb = y_tensor[idx]

                # empty categorical tensor (no categorical columns)
                xb_cat = torch.empty((len(idx), 0), dtype=torch.long, device=self.device)

                logits, loss = self.model(xb_cat, xb_num, yb)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            # ---------- validation / early stopping -------------------- #
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self._predict_tensor(X_val)
                    val_rmse = float(
                        torch.sqrt(
                            nn.MSELoss()(
                                torch.as_tensor(val_preds, device=self.device),
                                y_val,
                            )
                        ).item()
                    )

                if val_rmse < best_metric:
                    best_metric = val_rmse
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self.best_iteration = epoch
                    if patience_left is not None:
                        patience_left = self.early_stopping_rounds
                else:
                    if patience_left is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            if self.verbose:
                                print("Early stopping triggered.")
                            break

        # Reload best weights (if early stopping was used) ---------------
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _predict_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        preds = []
        for start in range(0, len(X_tensor), 8192):
            xb_num = X_tensor[start : start + 8192]
            b = xb_num.shape[0]
            xb_cat = torch.empty((b, 0), dtype=torch.long, device=self.device)
            dummy_label = torch.zeros(b, device=self.device)  # not used for inference
            out, _ = self.model(xb_cat, xb_num, dummy_label)
            preds.append(out.cpu())
        return torch.cat(preds).squeeze(-1).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)