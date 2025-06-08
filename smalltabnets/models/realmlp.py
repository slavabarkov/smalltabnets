import logging
import warnings
from datetime import timedelta
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

from .base import BaseTabularRegressor


def apply_fp16_monkey_patch():
    """
    Monkey patch for RealMLP_TD_Regressor to force fp16 precision for PyTorch Lightning Trainer.
    This patch modifies the NNAlgInterface.fit method to use mixed precision training.
    """

    from pytabkit.models import utils
    from pytabkit.models.alg_interfaces.base import SplitIdxs
    from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
    from pytabkit.models.training.lightning_modules import TabNNModule

    def patched_fit(
        self, ds, idxs_list, interface_resources, logger, tmp_folders, name
    ):
        # Original fit method of NNAlgInterface
        assert np.all(
            [
                idxs_list[i].train_idxs.shape[0] == idxs_list[0].train_idxs.shape[0]
                for i in range(len(idxs_list))
            ]
        )
        # we can then decompose the overall number of sub-splits into the number of splits
        # and the number of sub-splits per split

        # have the option to change the seeds (for comparing NNs with different random seeds)
        random_seed_offset = self.config.get("random_seed_offset", 0)
        if random_seed_offset != 0:
            idxs_list = [
                SplitIdxs(
                    train_idxs=idxs.train_idxs,
                    val_idxs=idxs.val_idxs,
                    test_idxs=idxs.test_idxs,
                    split_seed=idxs.split_seed + random_seed_offset,
                    sub_split_seeds=[
                        seed + random_seed_offset for seed in idxs.sub_split_seeds
                    ],
                    split_id=idxs.split_id,
                )
                for idxs in idxs_list
            ]

        # https://stackoverflow.com/questions/74364944/how-to-get-rid-of-info-logging-messages-in-pytorch-lightning
        log = logging.getLogger("lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)

        warnings.filterwarnings(
            "ignore",
            message="You defined a `validation_step` but have no `val_dataloader`.",
        )

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # to be safe wrt rounding errors, but might not be necessary
        )
        gpu_devices = interface_resources.gpu_devices
        self.device = gpu_devices[0] if len(gpu_devices) > 0 else "cpu"
        ds = ds.to(self.device)

        n_epochs = self.config.get("n_epochs", 256)
        self.model = TabNNModule(
            **utils.join_dicts({"n_epochs": 256, "logger": logger}, self.config),
            fit_params=self.fit_params,
        )
        self.model.compile_model(ds, idxs_list, interface_resources)

        if self.device == "cpu":
            pl_accelerator = "cpu"
            pl_devices = "auto"
        elif self.device == "mps":
            pl_accelerator = "mps"
            pl_devices = "auto"
        elif self.device == "cuda":
            pl_accelerator = "gpu"
            pl_devices = [0]
        elif self.device.startswith("cuda:"):
            pl_accelerator = "gpu"
            pl_devices = [int(self.device[len("cuda:") :])]
        else:
            raise ValueError(f'Unknown device "{self.device}"')

        max_time = (
            None
            if interface_resources.time_in_seconds is None
            else timedelta(seconds=interface_resources.time_in_seconds)
        )

        self.trainer = pl.Trainer(
            max_time=max_time,
            accelerator=pl_accelerator,
            devices=pl_devices,
            callbacks=self.model.create_callbacks(),
            max_epochs=n_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            logger=pl.loggers.logger.DummyLogger(),
            enable_model_summary=False,
            log_every_n_steps=1,
            precision="16-mixed",  # Use mixed precision training
        )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.model.train_dl,
            val_dataloaders=self.model.val_dl,
        )

        if hasattr(self.model, "fit_params"):
            self.fit_params = self.model.fit_params

        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        self.trainer.max_time = None

    # Apply the monkey patch
    NNAlgInterface.fit = patched_fit


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
        adam_beta_2: float = 0.999,  # Adam beta2
        # Embeddings
        use_embeddings: bool = True,
        embedding_type: str = "piecewise_linear",  # "piecewise_linear" or "linear"
        embedding_dim: int = 16,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        # Pass all base parameters to parent
        super().__init__(**kwargs)

        # RalMLP is a "bag-of-tricks"
        # For a fair comparison, we keep some parameters in line with other models

        # Store RealMLP-specific hyper-parameters
        self.n_cv = 1  # we provide our own validation set
        self.n_refit = 0
        self.val_fraction = 0  # always 0, we separate validation set manually

        # We preprocess data ourselves
        self.tfms = []

        # We process targets ourselves
        self.normalize_output = False  # default=True
        self.clamp_output = False  # default=True

        # Numerical embeddings
        self.num_emb_type = "none"
        self.use_plr_embeddings = False
        self.num_emb_dim = None
        if use_embeddings:
            self.tfms.append("embedding")
            if embedding_type == "piecewise_linear":
                self.num_emb_type = "pl"
                self.use_plr_embeddings = True
                self.num_emb_dim = embedding_dim

        # Early stopping
        self.early_stopping_multiplicative_patience = None
        self.early_stopping_additive_patience = None
        if self.use_early_stopping:
            self.early_stopping_multiplicative_patience = 1.0
            self.early_stopping_additive_patience = self.early_stopping_rounds

        # Paper architectural decisions
        self.act = "mish"
        self.use_parametric_act = True
        self.weight_param = "ntk"
        self.add_front_scale = True

        # Optimizer
        self.opt = "adamw"  # default='adam'
        self.adam_beta_2 = adam_beta_2  # Adam beta2
        self.lr_sched = "constant"  # default='coslog4'
        self.mom_sched = "constant"  # default='constant'
        self.sq_mom_sched = "constant"  # default='constant'
        self.opt_eps_sched = "constant"  # default='constant'
        self.wd_sched = "constant"
        self.p_drop_sched = "constant"  # default='flat_cos'

        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width
        self.hidden_sizes = "rectangular"  # [hidden_width] * n_hidden_layers
        self.wd = wd
        self.p_drop = p_drop

    def _create_model(self, n_features):
        # Monkey patch pytabkit to add AdamW support
        # The expected version of pytabkit for this monkey patch is 1.3.0

        # First, set up the AdamW optimizer class
        from pytabkit.models.optim.optimizers import OptimizerBase

        class AdamWOptimizer(OptimizerBase):
            def __init__(self, param_groups, hp_manager):
                super().__init__(
                    torch.optim.AdamW(param_groups),
                    hyper_mappings=[
                        ("lr", "lr", 1e-3),
                        (("mom", "sq_mom"), "betas", (0.9, 0.999)),
                        ("opt_eps", "eps", 1e-8),
                        (
                            "wd",
                            "weight_decay",
                            0.01,
                        ),  # AdamW uses weight_decay, not wd
                    ],
                    hp_manager=hp_manager,
                )

        # Monkey patch before importing anything else from pytabkit
        import pytabkit.models.optim.optimizers as optimizer_module

        # Store original get_opt_class
        _original_get_opt_class = optimizer_module.get_opt_class

        def patched_get_opt_class(opt_name):
            """Patched version that adds AdamW support"""
            if opt_name == "adamw":
                return AdamWOptimizer
            else:
                return _original_get_opt_class(opt_name)

        # Apply the monkey patch
        optimizer_module.get_opt_class = patched_get_opt_class

        # Also patch it in the lightning trainer module where it is imported
        try:
            import pytabkit.models.training.lightning_modules as lightning_modules

            if hasattr(lightning_modules, "get_opt_class"):
                lightning_modules.get_opt_class = patched_get_opt_class
        except:
            pass

        # Patch precision
        if self._autocast_enabled:
            apply_fp16_monkey_patch()

        # Now import RealMLP after patching
        from pytabkit import RealMLP_TD_Regressor

        # Create and return the RealMLP model with the specified parameters
        return RealMLP_TD_Regressor(
            n_cv=self.n_cv,
            n_refit=self.n_refit,
            val_fraction=self.val_fraction,
            tfms=self.tfms,
            normalize_output=self.normalize_output,
            clamp_output=self.clamp_output,
            num_emb_type=self.num_emb_type,
            use_plr_embeddings=self.use_plr_embeddings,
            plr_hidden_1=self.num_emb_dim,
            use_early_stopping=self.use_early_stopping,
            early_stopping_multiplicative_patience=self.early_stopping_multiplicative_patience,
            early_stopping_additive_patience=self.early_stopping_additive_patience,
            act=self.act,
            use_parametric_act=self.use_parametric_act,
            weight_param=self.weight_param,
            add_front_scale=self.add_front_scale,
            opt=self.opt,
            sq_mom=self.adam_beta_2,
            lr_sched=self.lr_sched,
            mom_sched=self.mom_sched,
            sq_mom_sched=self.sq_mom_sched,
            opt_eps_sched=self.opt_eps_sched,
            wd_sched=self.wd_sched,
            p_drop_sched=self.p_drop_sched,
            n_hidden_layers=self.n_hidden_layers,
            hidden_width=self.hidden_width,
            hidden_sizes=self.hidden_sizes,
            wd=self.wd,
            p_drop=self.p_drop,
            n_epochs=self.epochs,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            device=self.device,
            random_state=self.random_state,
            verbosity=self.verbose,
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
