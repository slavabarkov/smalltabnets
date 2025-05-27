from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..modules.modernnca.modernnca import ModernNCA
from .base import BaseTabularRegressor


class ModernNCARegressor(BaseTabularRegressor):
    """
    Wrapper around the ModernNCA model that provides the same
    unified interface as the other regressors in this repository.
    """

    def __init__(
        self,
        # Base parameters -------------------------------------------------
        epochs: int = 256,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        # Modern-NCA architecture ----------------------------------------
        dim: int = 128,
        d_block: int = 256,
        n_blocks: int = 0,
        dropout: float = 0.0,
        temperature: float = 1.0,
        sample_rate: float = 0.8,
        # Numerical‐embedding configuration (set to ``None`` for no embed)
        num_embeddings: Optional[dict] = None,
        d_num: int = 0,  # number of numerical features to embed
        # Misc ------------------------------------------------------------
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: int = 0,
        **kwargs,
    ):
        self.dim = dim
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.num_embeddings = num_embeddings
        self.d_num = d_num

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

    # --------------------------------------------------------------------- #
    #  Base-class hooks
    # --------------------------------------------------------------------- #
    def _get_expected_params(self):
        """
        Expected parameters for ModernNCA.  They are listed here so that
        they can be set through the YAML configuration files, if desired.
        """
        return [
            # training
            "epochs",
            "learning_rate",
            "batch_size",
            "use_early_stopping",
            "early_stopping_rounds",
            # architecture
            "dim",
            "d_block",
            "n_blocks",
            "dropout",
            "temperature",
            "sample_rate",
            "num_embeddings",
            "d_num",
            # misc
            "device",
            "random_state",
            "verbose",
        ]

    def _create_model(self, n_features: int):
        """
        Build the underlying ModernNCA instance.  For now we treat every
        feature as *raw* (i.e. no learnt embeddings).  If the user supplies
        a ``num_embeddings`` dictionary, ModernNCA will instead embed the
        ``d_num`` first features accordingly.
        """
        model = ModernNCA(
            d_in=n_features,
            d_num=self.d_num,
            d_out=1,  # regression
            dim=self.dim,
            dropout=self.dropout,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            num_embeddings=self.num_embeddings,
            temperature=self.temperature,
            sample_rate=self.sample_rate,
        )
        return model.to(self.device)

    # ------------------------------------------------------------------ #
    #  Training & inference
    # ------------------------------------------------------------------ #
    def _fit_model(self, X, y, eval_set=None):
        """
        Optimisation loop with optional early stopping.  The whole training
        set is used as the retrieval memory; ModernNCA performs its own
        random subsampling controlled by ``sample_rate``.
        """
        # Torch tensors
        X_train = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # Store retrieval memory for later prediction
        self._candidate_x = X_train
        self._candidate_y = y_train

        # Validation tensors if provided
        if eval_set and len(eval_set) > 0:
            X_val_np, y_val_np = eval_set[0]
            X_val = torch.as_tensor(X_val_np, dtype=torch.float32, device=self.device)
            y_val = torch.as_tensor(y_val_np, dtype=torch.float32, device=self.device)
        else:
            X_val = y_val = None

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_metric = float("inf")
        best_state = None
        patience_left = self.early_stopping_rounds if self.use_early_stopping else None
        self.best_iteration = self.epochs - 1  # will be overwritten

        rng = np.random.RandomState(self.random_state)

        for epoch in range(self.epochs):
            self.model.train()

            # Shuffled mini-batches
            indices = rng.permutation(X_train.shape[0])
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                preds = self.model(
                    xb,
                    yb,
                    candidate_x=self._candidate_x,
                    candidate_y=self._candidate_y,
                    is_train=True,
                )

                loss = criterion(preds, yb)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # ------------------ validation / early stopping --------------
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(
                        X_val,
                        None,
                        candidate_x=self._candidate_x,
                        candidate_y=self._candidate_y,
                        is_train=False,
                    )
                    val_rmse = torch.sqrt(criterion(val_preds, y_val)).item()

                if val_rmse < best_metric:
                    best_metric = val_rmse
                    self.best_iteration = epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    if patience_left is not None:
                        patience_left = self.early_stopping_rounds
                else:
                    if patience_left is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            if self.verbose:
                                print("Early stopping triggered.")
                            break

        # Reload best weights (if we used validation)
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------ #
    def _predict_model(self, X):
        """
        Forward pass for inference.  The retrieval memory is the original
        (pre-processed) training set stored during ``fit``.
        """
        if not hasattr(self, "_candidate_x"):
            raise RuntimeError("Model has to be fitted before calling predict.")

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), 8192):
                batch = X_tensor[start : start + 8192]
                out = self.model(
                    batch,
                    None,
                    candidate_x=self._candidate_x,
                    candidate_y=self._candidate_y,
                    is_train=False,
                )
                preds.append(out.cpu())

        return torch.cat(preds).numpy()
