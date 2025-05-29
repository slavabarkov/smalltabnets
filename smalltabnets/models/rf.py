from typing import Optional

from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from .base import BaseTabularRegressor


class RandomForestRegressor(BaseTabularRegressor):
    """Random Forest regressor with unified interface."""

    def __init__(
        self,
        # Base training parameters
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        # Base early stopping parameters
        use_early_stopping: Optional[bool] = None,
        early_stopping_rounds: Optional[int] = None,
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
        # RF specific parameters
        n_estimators: int = 1000,
        bootstrap: bool = True,
        criterion: str = "squared_error",
        n_jobs: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        super().__init__(
            # Base training parameters
            epochs=self.n_estimators,
            learning_rate=None,
            batch_size=None,
            # Base early stopping parameters
            use_early_stopping=None,  # RF does not use early stopping
            early_stopping_rounds=None,  # RF does not use early stopping
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
        return SklearnRandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def _fit_model(self, X, y, eval_set=None):
        # Random Forest doesn't use validation set
        self.model.fit(X, y)
        self.best_iteration = self.epochs  # Always uses all trees
