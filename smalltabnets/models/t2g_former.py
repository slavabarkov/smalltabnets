"""
Unified scikit-like wrapper around the T2G-Former architecture.

The class follows exactly the same contract as the other regressors in
`bodenboden/models` (`TabRRegressor`, `ModernNCARegressor`, …) so that it
can be plugged into `benchmark.py` without touching any other code.
"""

from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn

from ..modules.t2g_former.t2g_former import T2GFormer
from .base import BaseTabularRegressor


class T2GFormerRegressor(BaseTabularRegressor):
    """
    Wrapper that exposes T2G-Former through the unified
    `BaseTabularRegressor` interface.
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
        # >>> T2G-Former specific ---------------------------------------
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
        # >>> extra kwargs (ignored but accepted) ------------------------
        **kwargs,
    ):
        # Store T2G-Former parameters -----------------------------------
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

        # Let the base-class take care of the common bits ---------------
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
        Parameters that can appear in YAML / Optuna and should therefore be
        captured by `_get_model_params` (even though we don’t use it here
        directly, this enables transparent YAML mapping).
        """
        return [
            # optimisation ------------------------------------------------
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            # architecture -----------------------------------------------
            "token_bias",
            "n_layers",
            "d_token",
            "n_heads",
            "d_ffn_factor",
            "attention_dropout",
            "ffn_dropout",
            "residual_dropout",
            "activation",
            "prenormalization",
            "initialization",
            "kv_compression",
            "kv_compression_sharing",
            "sym_weight",
            "sym_topology",
            "nsi",
            # misc / bookkeeping -----------------------------------------
            "device",
            "random_state",
            "verbose",
            # preprocessing flags that may appear in YAML ----------------
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
        ]

    # ------------------------------------------------------------------ #
    def _create_model(self, n_features: int) -> nn.Module:
        """
        Instantiate the underlying T2G-Former.  All LimeSoDa features are
        numerical, so `categories=None` and `d_numerical = n_features`.
        """
        model = T2GFormer(
            # --- tokenizer ---------------------------------------------
            d_numerical=n_features,
            categories=None,  # purely numerical
            token_bias=self.token_bias,
            # --- transformer -------------------------------------------
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
            # --- Linformer style compression ---------------------------
            kv_compression=self.kv_compression,
            kv_compression_sharing=self.kv_compression_sharing,
            # --- Graph-estimator flags ---------------------------------
            sym_weight=self.sym_weight,
            sym_topology=self.sym_topology,
            nsi=self.nsi,
            # --- output -------------------------------------------------
            d_out=1,  # regression
        )
        return model.to(self.device)

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def _fit_model(self, X, y, eval_set=None):
        # Convert to tensors --------------------------------------------
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Validation data (if supplied) ---------------------------------
        if self.use_early_stopping and eval_set and len(eval_set) > 0:
            X_val_np, y_val_np = eval_set[0]
            X_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=self.device)
            patience_left = self.early_stopping_rounds
        else:
            X_val = y_val = None
            patience_left = None

        best_metric = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_iteration = self.epochs - 1  # overwritten if ES triggers

        # ------------------------------- optimisation loop ------------ #
        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(X_tensor))

            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                preds = self.model(xb, None)
                loss = criterion(preds, yb)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            # -------- validation / early stopping ---------------------- #
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self._predict_tensor(X_val)
                    val_rmse = float(
                        torch.sqrt(
                            criterion(
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

        # Reload best model (if we used validation / ES) ----------------
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
            xb = X_tensor[start : start + 8192]
            out = self.model(xb, None)
            preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
