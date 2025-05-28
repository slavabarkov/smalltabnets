from typing import Optional

import numpy as np
import rtdl_num_embeddings
import torch
import torch.nn as nn

from ..modules.modernnca.modernnca import ModernNCA
from .base import BaseTabularRegressor


class ModernNCARegressor(BaseTabularRegressor):
    def __init__(
        self,
        # Base parameters
        epochs: int = 256,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        use_early_stopping: bool = True,
        early_stopping_rounds: Optional[int] = 16,
        # Dimensionality reduction
        use_pca: bool = False,
        n_pca_components: Optional[int] = None,
        # Modern-NCA architecture
        dim: int = 128,
        d_block: int = 256,
        n_blocks: int = 0,
        dropout: float = 0.0,
        temperature: float = 1.0,
        sample_rate: float = 0.8,
        # Embeddings
        use_embeddings: bool = False,
        embedding_type: str = "piecewise_linear",  # "piecewise_linear" or "linear"
        embedding_dim: int = 16,
        # Misc
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

        # Embeddings
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_embeddings = None
        self.bins = None

        super().__init__(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            random_state=random_state,
            verbose=verbose,
            use_pca=use_pca,
            n_pca_components=n_pca_components,
            **kwargs,
        )

    def _get_expected_params(self):
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
        model = ModernNCA(
            d_in=n_features,
            d_num=n_features,
            d_out=1,  # regression
            dim=self.dim,
            dropout=self.dropout,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            num_embeddings=self.num_embeddings,
            bins=self.bins,
            temperature=self.temperature,
            sample_rate=self.sample_rate,
        )
        return model.to(self.device)

    def _prepare_embeddings(self, X):
        """
        Prepare numerical embeddings configuration.
        Subclasses can override for custom embedding logic.
        """
        if not self.use_embeddings:
            return

        if self.embedding_type == "piecewise_linear":
            self.bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(X))
            self.num_embeddings = {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": self.embedding_dim,
                "activation": False,
                "version": "B",
            }
        elif self.embedding_type == "linear":
            self.num_embeddings = {
                "type": "LinearEmbeddings",
                "d_embedding": self.embedding_dim,
            }
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def fit(self, X, y, eval_set=None, verbose=False):
        """Common fit interface overriding with embeddings init logic."""
        # Apply preprocessing
        X = self._apply_feature_preprocessing(X, fit=True)
        if self.use_embeddings:
            self._prepare_embeddings(X)

        y = self._apply_target_preprocessing(y, fit=True)
        eval_set = self._prepare_eval_set(eval_set)

        # Create model
        n_features = X.shape[1]
        self.model = self._create_model(n_features)

        # Fit model (subclass-specific)
        self._fit_model(X, y, eval_set)

        return self

    def _fit_model(self, X, y, eval_set=None):
        # Store retrieval memory for later prediction
        self._candidate_x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self._candidate_y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return self._fit_pytorch_model(X, y, eval_set)

    def _forward_pass(self, batch):
        xb = batch["inputs"]
        yb = batch["targets"]
        output = self.model(
            xb,
            yb,
            candidate_x=self._candidate_x,
            candidate_y=self._candidate_y,
            is_train=True,
        )
        return output

    def _predict_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        """ModernNCA-specific prediction with retrieval memory."""
        if not hasattr(self, "_candidate_x"):
            raise RuntimeError("Model must be fitted before calling predict.")

        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), 8192):
                batch = X_tensor[start : start + 8192]
                out = self.model(
                    batch,
                    None,  # No targets during inference
                    candidate_x=self._candidate_x,
                    candidate_y=self._candidate_y,
                    is_train=False,  # Inference mode
                )
                preds.append(out.cpu())
        return torch.cat(preds).numpy()

    def _predict_model(self, X):
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
