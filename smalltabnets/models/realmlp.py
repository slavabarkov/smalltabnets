import logging
from typing import Optional

import numpy as np
from pytabkit import RealMLP_TD_Regressor

from .base import BaseTabularRegressor


class RealMLPRegressor(BaseTabularRegressor):
    """RealMLP regressor with unified interface."""

    def __init__(
        self,
        *,
        # RealMLP specific parameters
        # Only store params we reference
        n_hidden_layers=3,
        hidden_width=256,
        wd=1e-2,
        p_drop=0.0,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        # Store RealMLP-specific hyper-parameters
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width
        self.val_fraction = 0  # always 0, we separate validation set manually
        self.hidden_sizes = [hidden_width] * n_hidden_layers
        self.wd = wd
        self.p_drop = p_drop

        # Pass all base parameters to parent
        super().__init__(**kwargs)

    def _create_model(self, n_features):
        return RealMLP_TD_Regressor(
            n_epochs=self.epochs,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            verbosity=self.verbose,
            n_hidden_layers=self.n_hidden_layers,
            hidden_width=self.hidden_width,
            hidden_sizes=self.hidden_sizes,
            wd=self.wd,
            p_drop=self.p_drop,
            val_fraction=self.val_fraction,
        )

    def _fit_model(self, X, y, eval_set=None):

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        if eval_set:
            # Concatenate train and validation
            X_val, y_val = eval_set[0]
            X_combined = np.vstack([X, X_val])
            y_combined = np.concatenate([y, y_val])
            val_idxs = np.arange(len(X), len(X_combined))

            self.model.fit(X_combined, y_combined, val_idxs=val_idxs)
        else:
            self.model.fit(X, y)

        # Set best iteration
        if hasattr(self.model, "alg_interface_"):
            stopped_list = self.model.alg_interface_.model.has_stopped_list.get(
                "rmse", []
            )
            if all(stopped_list):
                self.best_iteration = (
                    self.model.alg_interface_.model.best_mean_val_epochs["rmse"][0]
                )
            else:
                self.best_iteration = self.epochs
        else:
            self.best_iteration = self.epochs
