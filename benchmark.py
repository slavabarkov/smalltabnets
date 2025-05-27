import argparse
import json
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from importlib import import_module
import copy
import LimeSoDa
import numpy as np
import optuna
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def get_logger():
    """Return a logger that writes to file and console."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)

    log_fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    handlers = [
        logging.FileHandler(f"logs/run_{ts}.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=logging.INFO, format=log_fmt, handlers=handlers, force=True
    )  # reconfigure even in notebooks

    return logging.getLogger("benchmark")


log = get_logger()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to yaml config")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_limesoda_data(dataset_name, target):
    data = LimeSoDa.load_dataset(dataset_name)["Dataset"]
    all_target_cols = [c for c in data.columns if c.endswith("_target")]
    
    X = data.drop(columns=all_target_cols).fillna(0)
    y = data[target]

    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    return X, y


def sample_from_grid(grid, trial):
    sample = {}
    for hp, meta in grid.items():
        if meta["type"] == "int":
            sample[hp] = trial.suggest_int(hp, meta["low"], meta["high"])
        elif meta["type"] == "float":
            if meta.get("log", False):
                sample[hp] = trial.suggest_float(
                    hp, meta["low"], meta["high"], log=True
                )
            else:
                sample[hp] = trial.suggest_float(hp, meta["low"], meta["high"])
        elif meta["type"] == "categorical":
            sample[hp] = trial.suggest_categorical(hp, meta["choices"])
    return sample


def get_model(path_string, params):
    module_name, cls_name = path_string.split(".")
    mod = import_module(f"bodenboden.models.{module_name}")
    cls = getattr(mod, cls_name)
    return cls(**params)


def run(cfg):
    random_state = cfg.get("random_state", 42)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    out_dir = cfg["results_dir"]
    os.makedirs(out_dir, exist_ok=True)

    outer_k = cfg["optimization"]["cv"]["outer_folds"]
    inner_k = cfg["optimization"]["cv"]["inner_folds"]
    n_trials = cfg["optimization"]["n_trials"]

    for entry in cfg["datasets"]:
        ds, tgt = entry["dataset"], entry["target"]
        log.info(f"Starting dataset={ds}, target={tgt}")
        X, y = load_limesoda_data(ds, tgt)

        outer_split = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)

        # Parse strategies from config
        strategies = []
        strategies_params = {}

        for strategy_name, strategy_config in cfg["outer_train_strategies"].items():
            if strategy_config.get("enabled", True):
                strategies.append(strategy_name)
                strategies_params[strategy_name] = {
                    k: v for k, v in strategy_config.items() if k != "enabled"
                }

        outer_test_preds_by_strategy = {
            strategy: np.zeros_like(y, dtype=float) for strategy in strategies
        }

        fold_results = []  # list of dicts – one per outer fold

        for fold_out, (train_out_idx, test_out_idx) in enumerate(
            outer_split.split(X, y)
        ):
            X_out_train, y_out_train = X[train_out_idx], y[train_out_idx]
            X_out_test, y_out_test = X[test_out_idx], y[test_out_idx]

            def objective(trial):
                params = cfg["fixed_parameters"].copy()
                params.update(sample_from_grid(cfg["hyperparameter_grid"], trial))

                inner_split = KFold(
                    n_splits=inner_k, shuffle=True, random_state=random_state
                )
                fold_scores, best_rounds = [], []

                for train_inner_idx, val_inner_idx in inner_split.split(
                    X_out_train, y_out_train
                ):
                    X_inner_train, y_inner_train = (
                        X_out_train[train_inner_idx],
                        y_out_train[train_inner_idx],
                    )
                    X_inner_val, y_inner_val = (
                        X_out_train[val_inner_idx],
                        y_out_train[val_inner_idx],
                    )

                    model = get_model(cfg["model"], params)
                    model.fit(
                        X_inner_train,
                        y_inner_train,
                        eval_set=[(X_inner_val, y_inner_val)],
                        verbose=False,
                    )

                    y_pred_inner_val = model.predict(X_inner_val)
                    rmse_val = np.sqrt(
                        mean_squared_error(y_inner_val, y_pred_inner_val)
                    )
                    fold_scores.append(rmse_val)

                    if hasattr(model, "best_iteration"):
                        best_rounds.append(model.best_iteration + 1)

                trial.set_user_attr("best_rounds", best_rounds)
                return np.mean(fold_scores)

            study = None
            best_params = cfg["fixed_parameters"].copy()
            if n_trials > 0:
                sampler = optuna.samplers.TPESampler(
                    seed=random_state, **cfg["optimization"]["sampler_params"]
                )
                study = optuna.create_study(direction="minimize", sampler=sampler)

                study.optimize(
                    objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                )

                best_params.update(study.best_trial.params)
                log.info(f" Fold {fold_out}: best params={best_params}")

            # Now, we have best aprams - whether neede to tune them or not
            # Let's train outer model and save the predictions

            # Strategy: agg_epochs_mean
            # Aggregate the best rounds from all inner folds, and use the mean
            # as the number of epochs to train the outer model
            if "agg_epochs_mean" in strategies:
                strategy_best_params = copy.deepcopy(best_params)
                if study.best_trial.user_attrs["best_rounds"]:
                    rounds = np.mean(study.best_trial.user_attrs["best_rounds"])
                    rounds = int(rounds)
                    log.info(f"Strategy: mean rounds, training with {rounds} epochs")

                    if "n_estimators" in strategy_best_params:
                        strategy_best_params["n_estimators"] = rounds
                    elif "epochs" in strategy_best_params:
                        strategy_best_params["epochs"] = rounds
                    elif "n_epochs" in strategy_best_params:
                        strategy_best_params["n_epochs"] = rounds
                    strategy_best_params["early_stopping_rounds"] = None
                    strategy_best_params["use_early_stopping"] = False
                    model = get_model(cfg["model"], strategy_best_params)

                    # Fit with the whole outer training set
                    model.fit(X_out_train, y_out_train, verbose=False)
                    # Save fold/strategy predictions
                    outer_test_preds_by_strategy["agg_epochs_mean"][test_out_idx] = (
                        model.predict(X_out_test)
                    )

            # Strategy: agg_epochs_mean_std
            # Aggregate the best rounds from all inner folds, and use the mean
            # plus std as the number of epochs to train the outer model
            if "agg_epochs_mean_std" in strategies:
                strategy_best_params = copy.deepcopy(best_params)
                if study.best_trial.user_attrs["best_rounds"]:
                    rounds = np.mean(
                        study.best_trial.user_attrs["best_rounds"]
                    ) + np.std(study.best_trial.user_attrs["best_rounds"])
                    rounds = int(rounds)
                    log.info(
                        f"Strategy: mean+std rounds, training with {rounds} epochs"
                    )

                    if "n_estimators" in strategy_best_params:
                        strategy_best_params["n_estimators"] = rounds
                    elif "epochs" in strategy_best_params:
                        strategy_best_params["epochs"] = rounds
                    elif "n_epochs" in strategy_best_params:
                        strategy_best_params["n_epochs"] = rounds

                    strategy_best_params["early_stopping_rounds"] = None
                    model = get_model(cfg["model"], strategy_best_params)

                    # Fit with the whole outer training set
                    model.fit(X_out_train, y_out_train, verbose=False)
                    # Save fold/strategy predictions
                    outer_test_preds_by_strategy["agg_epochs_mean_std"][
                        test_out_idx
                    ] = model.predict(X_out_test)

            # Strategy: agg_epochs_percentile
            # Aggregate the best rounds from all inner folds, and use the 90th
            # percentile as the number of epochs to train the outer model
            if "agg_epochs_percentile" in strategies:
                rounds_percentile = strategies_params["agg_epochs_percentile"].get(
                    "percentile", 90
                )

                strategy_best_params = copy.deepcopy(best_params)
                if study.best_trial.user_attrs["best_rounds"]:
                    rounds = np.percentile(
                        study.best_trial.user_attrs["best_rounds"], rounds_percentile
                    )
                    rounds = int(rounds)
                    log.info(
                        f"Strategy: {rounds_percentile}th percentile rounds, training with {rounds} epochs"
                    )

                    if "n_estimators" in strategy_best_params:
                        strategy_best_params["n_estimators"] = rounds
                    elif "epochs" in strategy_best_params:
                        strategy_best_params["epochs"] = rounds
                    elif "n_epochs" in strategy_best_params:
                        strategy_best_params["n_epochs"] = rounds

                    strategy_best_params["early_stopping_rounds"] = None
                    model = get_model(cfg["model"], strategy_best_params)

                    # Fit with the whole outer training set
                    model.fit(X_out_train, y_out_train, verbose=False)
                    # Save fold/strategy predictions
                    outer_test_preds_by_strategy["agg_epochs_percentile"][
                        test_out_idx
                    ] = model.predict(X_out_test)

            # Strategy: full_train
            # Train the model with the whole outer training set, using 256 epochs
            if "full_train" in strategies:
                full_train_epochs = strategies_params["full_train"].get("epochs", 256)

                strategy_best_params = copy.deepcopy(best_params)
                log.info(
                    f"Strategy: full train, training with {full_train_epochs} epochs (no early stopping)"
                )
                if "n_estimators" in strategy_best_params:
                    strategy_best_params["n_estimators"] = full_train_epochs
                elif "epochs" in strategy_best_params:
                    strategy_best_params["epochs"] = full_train_epochs
                elif "n_epochs" in strategy_best_params:
                    strategy_best_params["n_epochs"] = full_train_epochs
                strategy_best_params["early_stopping_rounds"] = None
                model = get_model(cfg["model"], strategy_best_params)
                model.fit(X_out_train, y_out_train, verbose=False)
                outer_test_preds_by_strategy["full_train"][test_out_idx] = (
                    model.predict(X_out_test)
                )

            # Strategy: early_stopping
            # Use validation set, separated from outer training set
            # to determine the best number of epochs
            if "early_stopping" in strategies:
                early_stopping_val_frac = strategies_params["early_stopping"].get(
                    "val_frac", 0.2
                )

                strategy_best_params = copy.deepcopy(best_params)
                patience_pr = strategy_best_params.get("early_stopping_rounds", None)
                if patience_pr is None:
                    patience_pr = strategy_best_params.get(
                        "early_stopping_additive_patience", None
                    )
                log.info(
                    f"Strategy: early stopping, training with patience of {patience_pr}"
                )
                X_out_train_stop, X_out_val_stop, y_out_train_stop, y_out_val_stop = (
                    train_test_split(
                        X_out_train,
                        y_out_train,
                        test_size=early_stopping_val_frac,
                        random_state=random_state,
                        shuffle=True,
                    )
                )
                model = get_model(cfg["model"], strategy_best_params)
                model.fit(
                    X_out_train_stop,
                    y_out_train_stop,
                    eval_set=[(X_out_val_stop, y_out_val_stop)],
                    verbose=False,
                )
                outer_test_preds_by_strategy["early_stopping"][test_out_idx] = (
                    model.predict(X_out_test)
                )
            else:
                outer_test_preds_by_strategy.pop("early_stopping", None)

            # Strategy: ensemble inner models
            # Use the inner models to predict the outer test set
            if "ensemble" in strategies:
                strategy_best_params = copy.deepcopy(best_params)
                log.info(f"Strategy: ensemble inner models")
                inner_split = KFold(
                    n_splits=inner_k, shuffle=True, random_state=random_state
                )
                models_inner = []
                rmse_inner = []
                preds_outer = []
                for train_inner_idx, val_inner_idx in inner_split.split(
                    X_out_train, y_out_train
                ):
                    X_inner_train, y_inner_train = (
                        X_out_train[train_inner_idx],
                        y_out_train[train_inner_idx],
                    )
                    X_inner_val, y_inner_val = (
                        X_out_train[val_inner_idx],
                        y_out_train[val_inner_idx],
                    )

                    model = get_model(cfg["model"], strategy_best_params)
                    model.fit(
                        X_inner_train,
                        y_inner_train,
                        eval_set=[(X_inner_val, y_inner_val)],
                        verbose=False,
                    )
                    models_inner.append(model)
                    pred_inner = model.predict(X_inner_val)
                    rmse_inner.append(
                        np.sqrt(mean_squared_error(y_inner_val, pred_inner))
                    )
                    preds_outer.append(model.predict(X_out_test))
                preds_outer = np.array(preds_outer)
                preds_outer_mean = np.mean(preds_outer, axis=0)
                outer_test_preds_by_strategy["ensemble"][
                    test_out_idx
                ] = preds_outer_mean

            # Strategy: best_inner_model
            # Use the best inner model to predict the outer test set
            if "best_inner_model" in strategies:
                log.info(f"Strategy: best inner model")
                best_inner_model_idx = np.argmin(np.array(rmse_inner))
                preds_outer_best_inner = preds_outer[best_inner_model_idx]
                outer_test_preds_by_strategy["best_inner_model"][
                    test_out_idx
                ] = preds_outer_best_inner

            # Calculate outer fold metrics for each strategy and print
            for strategy, preds_fold in outer_test_preds_by_strategy.items():
                preds_fold = preds_fold[test_out_idx]
                fold_rmse = np.sqrt(mean_squared_error(y_out_test, preds_fold))
                fold_r2 = r2_score(y_out_test, preds_fold)
                log.info(
                    f"Fold {fold_out}: {strategy} RMSE={fold_rmse:.5f}   R²={fold_r2:.5f}"
                )

            fold_results.append(
                {
                    "outer_fold": fold_out,
                    "best_params": best_params,
                    "inner_best_rounds": (
                        study.best_trial.user_attrs["best_rounds"] if study else None
                    ),
                    "inner_cv_mse": study.best_value if study else None,
                    "outer_rmse": fold_rmse,
                    "outer_r2": fold_r2,
                }
            )

        # ───────── save predictions + best params ────────────────────────
        save_dir = os.path.join(out_dir, f'{ds}_{tgt.replace("_target","")}')
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(save_dir, "run_config_copy.yaml"))
        np.save(os.path.join(save_dir, "y_true.npy"), y)

        # Save predictions for each strategy
        for strategy, preds_all in outer_test_preds_by_strategy.items():
            np.save(os.path.join(save_dir, f"y_pred_{strategy}.npy"), preds_all)

        with open(os.path.join(save_dir, "best_params.json"), "w") as f:
            json.dump(fold_results, f, indent=2)

        # Print overall metrics for each strategy
        for strategy, preds_all in outer_test_preds_by_strategy.items():
            fold_rmse = np.sqrt(mean_squared_error(y, preds_all))
            fold_r2 = r2_score(y, preds_all)
            log.info(f"Overall: {strategy} RMSE={fold_rmse:.5f}   R²={fold_r2:.5f}")
        log.info(f"Finished {ds}/{tgt}")


# ──────────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config)
    run(cfg)
