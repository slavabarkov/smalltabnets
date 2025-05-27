import xgboost as xgb

from .base import BaseTabularRegressor


class XGBRegressor(BaseTabularRegressor):
    """XGBoost regressor with unified interface."""

    def _get_expected_params(self):
        return [
            "n_estimators",
            "learning_rate",
            "early_stopping_rounds",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "random_state",
            "n_jobs",
            "verbosity",
        ]

    def _create_model(self, n_features):
        params = self._get_model_params()

        # XGBoost doesn't use explicit 'use_early_stopping' as a parameter
        # but handles it implicitly when 'early_stopping_rounds' is provided
        if not self.use_early_stopping:
            params.pop("early_stopping_rounds", None)

        return xgb.XGBRegressor(**params)

    def _fit_model(self, X, y, eval_set=None):
        # XGBoost handles early stopping internally
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=self.model.verbosity > 0,
        )

        # Set best iteration
        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration
        else:
            self.best_iteration = self.epochs
