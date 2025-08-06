import rtdl_num_embeddings
import torch

from ..modules.realmlp.realmlp import RealMLPRegressor as RealMLPModule
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
        p_drop=0.0,
        # Embeddings
        use_embeddings: bool = True,
        embedding_type: str = "piecewise_linear",
        embedding_dim: int = 16,
        # Accept all base parameters via **kwargs
        **kwargs,
    ):
        # Pass all base parameters to parent
        super().__init__(**kwargs)

        self.hidden_sizes = [hidden_width] * n_hidden_layers
        self.p_drop = p_drop

        # Embeddings
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.bins = None

    def _prepare_embeddings(self, X):
        """
        Prepare numerical embeddings configuration.
        Similar to TabM implementation.
        """
        if not self.use_embeddings:
            return

        if self.embedding_type == "piecewise_linear":
            # Use heuristic
            # We set a maximum of 48 bins, and a minimum of len(X) // 2 bins.
            n_bins = min(48, len(X) // 2)
            # Compute bins using quantiles
            self.bins = rtdl_num_embeddings.compute_bins(
                torch.as_tensor(X),
                n_bins=n_bins,
            )
        else:
            raise NotImplementedError(f"Unknown embedding type: {self.embedding_type}")

    def _create_model(self, n_features):
        """Create RealMLP model instance."""
        model = RealMLPModule(
            in_features=n_features,
            hidden_sizes=self.hidden_sizes,
            dropout=self.p_drop,
            use_front_scale=True,
            use_parametric_act=True,
            use_num_emb=self.use_embeddings,
            num_emb_type=self.embedding_type,
            num_emb_dim=self.embedding_dim,
            num_emb_bins=self.bins,
            out_features=1,
        )
        return model.to(self.device)

    def _forward_pass(self, batch):
        """RealMLP forward pass."""
        return self.model(batch["inputs"]).squeeze(-1)

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
        """Use common PyTorch training loop."""
        self._fit_pytorch_model(X, y, eval_set)

    def _predict_model(self, X):
        """Make predictions with the underlying model."""
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(X_tensor)
