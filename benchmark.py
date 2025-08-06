import argparse
import copy
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path

import LimeSoDa
import numpy as np
import optuna
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def get_logger(
    log_to_stdout: bool = True,
    log_to_file: bool = False,
) -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)

    handlers = []
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_to_file:
        handlers.append(logging.FileHandler(f"logs/run_{ts}.log", mode="w"))

    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        force=True,
    )
    return logging.getLogger("benchmark")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to yaml config")
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
    mod = import_module(f"smalltabnets.models.{module_name}")
    cls = getattr(mod, cls_name)
    return cls(**params)


def load_best_params_from_file(json_path: Path):
    recs = json.loads(Path(json_path).read_text())
    # turn list of dicts –> dict[outer_fold] = rec
    return {rec["outer_fold"]: rec for rec in recs}


def run(cfg):
    random_state_base = cfg.get("random_state", 42)
    torch.manual_seed(random_state_base)

    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_k = cfg["optimization"]["cv"]["outer_folds"]
    inner_k = cfg["optimization"]["cv"]["inner_folds"]
    n_trials = int(cfg["optimization"]["n_trials"])

    # Will hold pre-tuned params if they are provided in YAML config
    best_params_source = cfg["optimization"].get("best_params")
    cached_best_params: dict[str, dict[int, dict]] = {}  # (ds_tgt) -> fold -> rec

    for entry in cfg["datasets"]:
        ds, tgt = entry["dataset"], entry["target"]
        ds_tag = f'{ds}_{tgt.replace("_target","")}'
        log.info(f"Starting dataset={ds}, target={tgt}")
        X, y = load_limesoda_data(ds, tgt)

        outer_split = KFold(
            n_splits=outer_k,
            shuffle=True,
            random_state=random_state_base,
        )

        # Parse strategies from config
        strategies, strategies_params = [], {}
        for name, s_cfg in cfg["outer_train_strategies"].items():
            if not s_cfg.get("enabled", True):
                continue
            strategies.append(name)
            strategies_params[name] = {k: v for k, v in s_cfg.items() if k != "enabled"}
        log.info(f"Active strategies: {strategies}")

        # If we have pre-tuned params, load them
        if n_trials == 0 and best_params_source:
            if os.path.isdir(best_params_source):
                bp_file = Path(best_params_source) / ds_tag / "best_params.json"
            else:
                bp_file = Path(best_params_source)
            if bp_file.exists():
                cached_best_params[ds_tag] = load_best_params_from_file(bp_file)
                log.info(f"Loaded cached best params from {bp_file}")
            else:
                raise FileNotFoundError(
                    f"n_trials==0 but no best_param file found at {bp_file}"
                )

        outer_test_preds_by_strategy = dict()
        for strategy in strategies:
            if "ensemble" not in strategy:
                outer_test_preds_by_strategy[strategy] = np.empty_like(y, dtype=float)

            if "ensemble" in strategy:
                # If ensembles are in strategies, we read the k from config
                # and create array of shape (k, y)
                ens_cfg = strategies_params[strategy]
                k = int(ens_cfg.get("k", 5))
                outer_test_preds_by_strategy[strategy] = np.empty(
                    (k, len(y)), dtype=float
                )

        fold_records = []  # list of dicts – one per outer fold

        for fold_out, (train_out_idx, test_out_idx) in enumerate(
            outer_split.split(X, y)
        ):
            X_out_train, y_out_train = X[train_out_idx], y[train_out_idx]
            X_out_test, y_out_test = X[test_out_idx], y[test_out_idx]

            # =============== hyper-parameter tuning or reuse ===================
            inner_best_rounds: list[int] | None = None
            best_params = copy.deepcopy(cfg["fixed_parameters"])
            study = None

            if n_trials == 0 and ds_tag not in cached_best_params:
                log.info(f"No tuned parameters were provided, using defaults.")
                pass

            if n_trials == 0 and ds_tag in cached_best_params:
                rec = cached_best_params[ds_tag][fold_out]
                best_params.update(rec["best_params"])
                inner_best_rounds = rec.get("inner_best_rounds", None)

            if n_trials > 0:

                def objective(trial):
                    params = cfg["fixed_parameters"].copy()
                    params.update(sample_from_grid(cfg["hyperparameter_grid"], trial))

                    inner_split = KFold(
                        n_splits=inner_k,
                        shuffle=True,
                        random_state=random_state_base,
                    )
                    rmse_scores, best_rounds = [], []

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
                        rmse_scores.append(rmse_val)

                        if hasattr(model, "best_iteration"):
                            best_rounds.append(model.best_iteration + 1)

                    trial.set_user_attr("best_rounds", best_rounds)
                    return np.mean(rmse_scores)

                sampler_type = cfg["optimization"].get("sampler", "tpe").lower()
                if sampler_type == "bruteforce":
                    sampler = optuna.samplers.BruteForceSampler(
                        seed=random_state_base,
                        **cfg["optimization"].get("sampler_params", {}),
                    )
                elif sampler_type == "tpe":
                    sampler = optuna.samplers.TPESampler(
                        seed=random_state_base,
                        **cfg["optimization"].get("sampler_params", {}),
                    )

                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                )
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    show_progress_bar=False,
                )

                best_params.update(study.best_trial.params)
                inner_best_rounds = study.best_trial.user_attrs["best_rounds"]
                log.info(f" Fold {fold_out}: best params={best_params}")

            # Now, we have best params - whether we needed to tune them or not
            # Let's train outer model and save the predictions

            # =============== aggregation strategies ===================
            # (1) agg_epochs_mean
            if "agg_epochs_mean" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)
                    mean_epochs = (
                        np.mean(inner_best_rounds) if inner_best_rounds else None
                    )
                    if mean_epochs is not None:
                        # We are interested in n_estimators or epochs
                        if "n_estimators" in params_strategy:
                            params_strategy["n_estimators"] = int(mean_epochs)
                        elif "epochs" in params_strategy:
                            params_strategy["epochs"] = int(mean_epochs)

                    # Disable early stopping
                    params_strategy["use_early_stopping"] = False
                    params_strategy["early_stopping_rounds"] = None

                    model = get_model(cfg["model"], params_strategy)
                    model.fit(X_out_train, y_out_train, verbose=False)
                    preds = model.predict(X_out_test)
                    outer_test_preds_by_strategy["agg_epochs_mean"][
                        test_out_idx
                    ] = preds

            # (2) agg_epochs_mean_ensemble
            if "agg_epochs_mean_ensemble" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)
                    mean_epochs = (
                        np.mean(inner_best_rounds) if inner_best_rounds else None
                    )
                    if mean_epochs is not None:
                        # We are interested in n_estimators or epochs
                        if "n_estimators" in params_strategy:
                            params_strategy["n_estimators"] = int(mean_epochs)
                        elif "epochs" in params_strategy:
                            params_strategy["epochs"] = int(mean_epochs)

                    # Disable early stopping
                    params_strategy["use_early_stopping"] = False
                    params_strategy["early_stopping_rounds"] = None

                    ens_cfg = strategies_params["agg_epochs_mean_ensemble"]
                    k = int(ens_cfg.get("k", 5))

                    preds_k = []
                    for seed in range(k):
                        params_ens_member = copy.deepcopy(params_strategy)
                        params_ens_member["random_state"] = seed

                        model = get_model(cfg["model"], params_ens_member)
                        model.fit(X_out_train, y_out_train, verbose=False)
                        preds_k.append(model.predict(X_out_test))

                    outer_test_preds_by_strategy["agg_epochs_mean_ensemble"][
                        :, test_out_idx
                    ] = preds_k

            # (3) early_stopping
            if "early_stopping" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)

                    es_cfg = strategies_params["early_stopping"]
                    val_frac = es_cfg.get("val_frac", 0.2)

                    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                        X_out_train,
                        y_out_train,
                        test_size=val_frac,
                        random_state=random_state_base,
                        shuffle=True,
                    )

                    model = get_model(cfg["model"], params_strategy)
                    model.fit(
                        X_train_es,
                        y_train_es,
                        eval_set=[(X_val_es, y_val_es)],
                        verbose=False,
                    )
                    preds = model.predict(X_out_test)
                    outer_test_preds_by_strategy["early_stopping"][test_out_idx] = preds

            # (4) early_stopping_ensemble
            if "early_stopping_ensemble" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)

                    es_cfg = strategies_params["early_stopping_ensemble"]
                    val_frac = es_cfg.get("val_frac", 0.2)
                    k = int(es_cfg.get("k", 5))

                    preds_k = []
                    for seed in range(k):
                        params_ens_member = copy.deepcopy(params_strategy)
                        params_ens_member["random_state"] = seed

                        X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                            X_out_train,
                            y_out_train,
                            test_size=val_frac,
                            random_state=seed,
                            shuffle=True,
                        )

                        model = get_model(cfg["model"], params_ens_member)
                        model.fit(
                            X_train_es,
                            y_train_es,
                            eval_set=[(X_val_es, y_val_es)],
                            verbose=False,
                        )
                        preds_k.append(model.predict(X_out_test))

                    outer_test_preds_by_strategy["early_stopping_ensemble"][
                        :, test_out_idx
                    ] = preds_k

            # (5) full_eps
            if "full_eps" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)

                    strat_cfg = strategies_params["full_eps"]
                    val_frac = strat_cfg.get("val_frac", 0.2)
                    n_full_eps = int(strat_cfg.get("epochs", 256))

                    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                        X_out_train,
                        y_out_train,
                        test_size=val_frac,
                        random_state=random_state_base,
                        shuffle=True,
                    )

                    # Override eps
                    params_strategy["early_stopping_rounds"] = n_full_eps
                    params_strategy["epochs"] = n_full_eps

                    model = get_model(cfg["model"], params_strategy)
                    model.fit(
                        X_train_es,
                        y_train_es,
                        eval_set=[(X_val_es, y_val_es)],
                        verbose=False,
                    )
                    preds = model.predict(X_out_test)
                    outer_test_preds_by_strategy["full_eps"][test_out_idx] = preds

            # (6) full_eps_ensemble
            if "full_eps_ensemble" in strategies:
                with torch.random.fork_rng():
                    torch.manual_seed(random_state_base)

                    params_strategy = copy.deepcopy(best_params)

                    strat_cfg = strategies_params["full_eps_ensemble"]
                    es_cfg = strategies_params["full_eps_ensemble"]
                    val_frac = es_cfg.get("val_frac", 0.2)
                    n_full_eps = int(strat_cfg.get("epochs", 256))

                    k = int(es_cfg.get("k", 5))

                    # Override eps
                    params_strategy["early_stopping_rounds"] = n_full_eps
                    params_strategy["epochs"] = n_full_eps

                    preds_k = []
                    for seed in range(k):
                        params_ens_member = copy.deepcopy(params_strategy)
                        params_ens_member["random_state"] = seed

                        X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
                            X_out_train,
                            y_out_train,
                            test_size=val_frac,
                            random_state=seed,
                            shuffle=True,
                        )

                        model = get_model(cfg["model"], params_ens_member)
                        model.fit(
                            X_train_es,
                            y_train_es,
                            eval_set=[(X_val_es, y_val_es)],
                            verbose=False,
                        )
                        preds_k.append(model.predict(X_out_test))

                    outer_test_preds_by_strategy["full_eps_ensemble"][
                        :, test_out_idx
                    ] = preds_k

            # Calculate outer fold metrics for each strategy and print
            for strat in strategies:
                if "ensemble" in strat:
                    # For ensemble strategies, we have k predictions
                    preds = outer_test_preds_by_strategy[strat][:, test_out_idx]
                    preds = np.mean(preds, axis=0)  # average over k predictions
                else:
                    # For non-ensemble strategies, we have a single prediction
                    preds = outer_test_preds_by_strategy[strat][test_out_idx]

                fold_rmse = np.sqrt(mean_squared_error(y_out_test, preds))
                fold_r2 = r2_score(y_out_test, preds)
                log.info(
                    f"Fold {fold_out} – {strat:27s}  "
                    f"RMSE={fold_rmse:.5f}  R²={fold_r2:.5f}"
                )

            fold_records.append(
                dict(
                    outer_fold=fold_out,
                    best_params=best_params,
                    inner_best_rounds=inner_best_rounds,
                    inner_cv_mse=(study.best_value if n_trials else None),
                )
            )

        # =============== save results for this dataset ===================
        save_dir = out_dir / ds_tag
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, save_dir / "run_config_copy.yaml")
        np.save(save_dir / "y_true.npy", y)

        for strat, preds in outer_test_preds_by_strategy.items():
            np.save(save_dir / f"y_pred_{strat}.npy", preds)

        (save_dir / "best_params.json").write_text(json.dumps(fold_records, indent=2))

        # overall metrics
        for strat, preds in outer_test_preds_by_strategy.items():
            if "ensemble" in strat:
                # For ensemble strategies, we have k predictions
                preds = np.mean(preds, axis=0)

            rmse = np.sqrt(mean_squared_error(y, preds))
            r2 = r2_score(y, preds)
            log.info(f"Overall -- {strat:27s}  RMSE={rmse:.5f}  R²={r2:.5f}")

        log.info(f" Finished {ds}/{tgt}, results saved to {save_dir}")


if __name__ == "__main__":
    log = get_logger()

    args = parse_args()
    cfg = load_cfg(args.config)
    run(cfg)
