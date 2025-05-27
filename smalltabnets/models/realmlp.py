import numpy as np
from pytabkit import RealMLP_TD_Regressor

from .base import BaseTabularRegressor


class RealMLPRegressor(BaseTabularRegressor):
    """RealMLP regressor with unified interface."""

    def __init__(
        self,
        # Base parameters
        epochs=256,
        learning_rate=0.2,
        batch_size=256,
        use_early_stopping=False,
        early_stopping_rounds=None,
        # Only store params we need to reference
        n_hidden_layers=3,
        hidden_width=256,
        # Misc
        device=None,
        random_state=42,
        verbose=0,
        **kwargs
    ):

        # Only store parameters we need to reference later
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width

        # All other RealMLP parameters will be passed through kwargs
        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

    def _get_expected_params(self):
        # All parameters that RealMLP accepts
        return [
            "n_epochs",
            "lr",
            "early_stopping_additive_patience",
            "early_stopping_multiplicative_patience",
            "batch_size",
            "lr_sched",
            "opt",
            "p_drop",
            "p_drop_sched",
            "wd",
            "wd_sched",
            "device",
            "random_state",
            "verbosity",
            "use_early_stopping",
            "n_hidden_layers",
            "hidden_width",
            "hidden_sizes",
            "val_fraction",
            "share_training_batches",
            "beta1",
            "beta2",
            "momentum",
            "weight_decay",
            "l1_reg",
            "l2_reg",
            "gradient_clipping_norm",
            "swa",
            "swa_start_epoch",
            "swa_lr",
        ]

    def _create_model(self, n_features):
        # Get all parameters (from kwargs, mappings, and attributes)
        params = self._get_model_params()

        # Override only the parameters we need to control
        params["val_fraction"] = 0  # Always 0 as required

        # Add hidden_sizes if not already present
        if "hidden_sizes" not in params and self.n_hidden_layers and self.hidden_width:
            params["hidden_sizes"] = [self.hidden_width] * self.n_hidden_layers

        return RealMLP_TD_Regressor(**params)

    def _fit_model(self, X, y, eval_set=None):
        import logging

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
