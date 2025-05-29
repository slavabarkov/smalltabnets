import logging
from typing import Optional

import numpy as np
from pytabkit import RealMLP_TD_Regressor

from .base import BaseTabularRegressor


class RealMLPRegressor(BaseTabularRegressor):
    """RealMLP regressor with unified interface."""

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
        # RealMLP specific parameters
        # Only store params we reference
        n_hidden_layers=3,
        hidden_width=256,
        wd=1e-2,
        p_drop=0.0,
    ):
        # Store RealMLP-specific hyper-parameters
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width
        self.val_fraction = 0  # always 0, we separate validation set manually
        self.hidden_sizes = [hidden_width] * n_hidden_layers
        self.wd = wd
        self.p_drop = p_drop

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
