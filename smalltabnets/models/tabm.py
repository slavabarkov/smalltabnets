import math
from typing import Optional

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn

from ..modules.tabm import TabMModel, make_parameter_groups
from .base import BaseTabularRegressor


class TabMRegressor(BaseTabularRegressor):
    """TabM regressor with unified interface."""

    def __init__(
        self,
        # Base parameters
        epochs=1000,
        learning_rate=2e-3,
        batch_size=16,
        use_early_stopping=True,
        early_stopping_rounds=16,
        # Architecture parameters
        arch_type: str = "tabm",
        k: int = 32,
        # Parameters for building the backbone
        n_blocks=3,
        d_block=512,
        dropout=0.1,
        activation="ReLU",
        # Training parameters
        weight_decay=3e-4,
        beta1=0.9,
        beta2=0.999,
        share_training_batches: bool = True,
        gradient_clipping_norm: Optional[float] = 1.0,
        # Embeddings
        piecewise_linear_embeddings: bool = False,
        # Misc
        device=None,
        random_state=42,
        verbose=0,
        **kwargs,
    ):
        self.arch_type = arch_type
        self.k = k

        self.piecewise_linear_embeddings = piecewise_linear_embeddings
        self.num_embeddings = None
        self.bins = None

        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.activation = activation

        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_clipping_norm = gradient_clipping_norm
        self.share_training_batches = share_training_batches

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

    def _get_expected_params(self):
        return [
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            "k",
            "arch_type",
            "n_blocks",
            "d_block",
            "dropout",
            "activation",
            "weight_decay",
            "beta1",
            "beta2",
            "gradient_clipping_norm",
            "feature_scaling",
            "standardize_targets",
            "clip_features",
            "clip_outputs",
            "device",
            "random_state",
            "verbose",
            "piecewise_linear_embeddings",
        ]

    def _create_model(self, n_features):
        model = TabMModel(
            n_num_features=n_features,
            cat_cardinalities=[],
            n_classes=None,  # regression
            backbone={
                "type": "MLP",
                "n_blocks": self.n_blocks,
                "d_block": self.d_block,
                "dropout": self.dropout,
                "activation": self.activation,
            },
            bins=self.bins,
            num_embeddings=self.num_embeddings,
            arch_type=self.arch_type,
            k=self.k,
            share_training_batches=self.share_training_batches,
        )

        return model.to(self.device)

    def fit(self, X, y, eval_set=None, verbose=False):
        """Common fit interface."""
        # Apply preprocessing
        X = self._apply_feature_preprocessing(X, fit=True)

        if self.piecewise_linear_embeddings:
            self.bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(X))
            self.num_embeddings = {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": 16,
                "activation": False,
                "version": "B",
            }

        y = self._apply_target_preprocessing(y, fit=True)
        eval_set = self._prepare_eval_set(eval_set)

        # Create model
        n_features = X.shape[1]
        self.model = self._create_model(n_features)

        # Fit model (subclass-specific)
        self._fit_model(X, y, eval_set)

    def _fit_model(self, X, y, eval_set=None):
        # Convert to tensors
        X_tensor = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, device=self.device, dtype=torch.float32)

        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)

        # Setup optimizer
        optim = torch.optim.AdamW(
            make_parameter_groups(self.model),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )
        criterion = nn.MSELoss()

        # Validation setup
        if self.use_early_stopping:
            X_val, y_val = eval_set[0]
            X_val = torch.as_tensor(X_val, device=self.device, dtype=torch.float32)
            y_val = torch.as_tensor(y_val, device=self.device, dtype=torch.float32)

        # Training loop
        best_metric = math.inf
        patience_left = self.early_stopping_rounds if self.use_early_stopping else 0
        self.best_iteration = None
        best_state = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            perm = rng.permutation(len(X_tensor))

            for start in range(0, len(X_tensor), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                optim.zero_grad()
                out = self.model(xb).squeeze(-1)

                if self.k:
                    loss = criterion(out.flatten(), yb.repeat_interleave(self.k))
                else:
                    loss = criterion(out.squeeze(-1), yb)

                loss.backward()

                if self.gradient_clipping_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping_norm
                    )

                optim.step()

            # Validation and early stopping
            if self.use_early_stopping:
                val_pred = self._predict_model(X_val)
                val_rmse = self._rmse(val_pred, y_val.cpu().numpy())

                if val_rmse < best_metric:
                    best_metric = val_rmse
                    self.best_iteration = epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    patience_left = self.early_stopping_rounds
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose > 0:
                            print("Early stopping!")
                        break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif self.best_iteration is None:
            self.best_iteration = self.epochs - 1

    def _rmse(self, pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            # Predict in batches
            preds = []
            for start in range(0, len(X_tensor), 8192):
                batch = X_tensor[start : start + 8192]
                out = self.model(batch).squeeze(-1).mean(1)
                preds.append(out.cpu())

        return torch.cat(preds).numpy()
