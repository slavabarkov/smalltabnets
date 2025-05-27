import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseTabularRegressor
from ..modules.tabr.tabr import TabR


class TabRRegressor(BaseTabularRegressor):
    """
    Wrapper that exposes the TabR model through the unified interface used
    in this repository (BaseTabularRegressor).  All common functionality
    such as feature-scaling, standardisation, clipping, etc. is inherited
    from the base-class – here we only need to

    1. declare which hyper-parameters TabR understands;
    2. build / train / predict with the underlying TabR network.
    """

    # ------------------------------------------------------------------ #
    #  Construction & hyper-parameters
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        # Base (shared) parameters ---------------------------------------
        epochs: int = 256,
        learning_rate: float = 2e-3,
        batch_size: int = 32,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        # TabR specific ---------------------------------------------------
        d_main: int = 64,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 2,
        predictor_n_blocks: int = 2,
        mixer_normalization: str = "auto",
        context_dropout: float = 0.1,
        dropout0: float = 0.1,
        dropout1: str = "dropout0",  # may be "dropout0" as in paper
        normalization: str = "LayerNorm",
        activation: str = "ReLU",
        context_size: int = 16,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
        # Misc ------------------------------------------------------------
        device: Optional[str] = "cuda",
        random_state: int = 42,
        verbose: int = 0,
        **kwargs,
    ):
        # Store TabR specific parameters
        self.d_main = d_main
        self.d_multiplier = d_multiplier
        self.encoder_n_blocks = encoder_n_blocks
        self.predictor_n_blocks = predictor_n_blocks
        self.mixer_normalization = mixer_normalization
        self.context_dropout = context_dropout
        self.dropout0 = dropout0
        self.dropout1 = dropout1
        self.normalization = normalization
        self.activation = activation
        self.context_size = context_size
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size

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
    def _get_expected_params(self):
        """
        All parameters that may be provided via YAML / Optuna and should be
        forwarded to TabR’s constructor.
        """
        return [
            # Base / optimisation
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            #  TabR architecture
            "d_main",
            "d_multiplier",
            "encoder_n_blocks",
            "predictor_n_blocks",
            "mixer_normalization",
            "context_dropout",
            "dropout0",
            "dropout1",
            "normalization",
            "activation",
            "context_size",
            "memory_efficient",
            "candidate_encoding_batch_size",
            # Misc
            "device",
            "random_state",
            "verbose",
            # preprocessing flags (handled by base but allowed in YAML)
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
        ]

    # ------------------------------------------------------------------ #
    def _create_model(self, n_features: int):
        """
        Instantiate TabR.  All features in LimeSoDa are treated as numerical
        (n_num_features = n_features, n_cat_features = 0).
        """
        model = TabR(
            # data-related
            n_num_features=n_features,
            n_cat_features=0,
            n_classes=1,  # regression
            # embeddings (none for purely numerical data)
            num_embeddings=None,
            # architecture / training params
            d_main=self.d_main,
            d_multiplier=self.d_multiplier,
            encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks,
            mixer_normalization=self.mixer_normalization,
            context_dropout=self.context_dropout,
            dropout0=self.dropout0,
            dropout1=self.dropout1,
            normalization=self.normalization,
            activation=self.activation,
            memory_efficient=self.memory_efficient,
            candidate_encoding_batch_size=self.candidate_encoding_batch_size,
        )
        return model.to(self.device)

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def _fit_model(self, X, y, eval_set=None):
        """
        Simple SGD training loop with optional early stopping.
        For each mini-batch we supply the rest of the training data as the
        retrieval memory (candidates) as required by TabR.
        """
        # Torch tensors ---------------------------------------------------
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        n_train = X_tensor.shape[0]
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        # Optimiser / loss
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Validation tensors (if given)
        if self.use_early_stopping and eval_set is not None:
            X_val_np, y_val_np = eval_set[0]
            X_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=self.device)
        else:
            X_val = y_val = None

        # Store candidate memory for later inference
        self._candidate_x_num = X_tensor
        self._candidate_y = y_tensor
        self._candidate_x_cat = None  # no categorical features

        # Early-stopping bookkeeping
        best_metric = math.inf
        best_state = None
        patience_left = (
            self.early_stopping_rounds
            if (self.use_early_stopping and X_val is not None)
            else None
        )
        self.best_iteration = self.epochs - 1  # will be overwritten if ES triggers

        # ----------------------------------------------------------------
        for epoch in range(self.epochs):
            self.model.train()

            # Shuffle mini-batches
            indices = rng.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_tensor[batch_idx]
                yb = y_tensor[batch_idx]

                # Build candidate memory WITHOUT current batch
                mask = torch.ones(n_train, dtype=torch.bool, device=self.device)
                mask[batch_idx] = False
                candidate_x_num = X_tensor[mask]
                candidate_y = y_tensor[mask]

                preds = self.model(
                    x_num=xb,
                    x_cat=None,
                    y=yb,
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=None,
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=True,
                ).squeeze(-1)

                loss = criterion(preds, yb)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            # ---------------- validation / early stopping ---------------
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

        # Reload best weights (if we used validation / ES)
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #
    def _predict_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        """
        Internal helper that assumes pre-processed *tensor* input and returns
        a NumPy array on CPU.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], 8192):
                xb = X_tensor[start : start + 8192]
                out = self.model(
                    x_num=xb,
                    x_cat=None,
                    y=None,
                    candidate_x_num=self._candidate_x_num,
                    candidate_x_cat=None,
                    candidate_y=self._candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
