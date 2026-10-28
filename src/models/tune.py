"""Hyperparameter tuning via Optuna with temporal validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.train import apply_smote, train_xgboost, _prepare_xgboost_params
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_TUNING_METRICS = {
    "average_precision": average_precision_score,
    "pr_auc": average_precision_score,
    "aucpr": average_precision_score,
    "roc_auc": roc_auc_score,
}


def _score_predictions(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Evaluate a tuning metric from validation labels and probabilities."""
    metric_key = metric_name.strip().lower()
    metric_fn = SUPPORTED_TUNING_METRICS.get(metric_key)
    if metric_fn is None:
        raise ValueError(
            f"Unsupported Optuna metric '{metric_name}'. "
            f"Choose from {sorted(SUPPORTED_TUNING_METRICS)}."
        )
    return float(metric_fn(y_true, y_prob))


def _resolve_storage_url(
    storage_url: str | None,
    study_name: str,
    experiments_dir: Path,
) -> str:
    """Resolve study storage to a persistent absolute URL when possible."""
    if not storage_url:
        default_path = experiments_dir / f"{study_name}.db"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{default_path.resolve().as_posix()}"

    if not storage_url.startswith("sqlite:///"):
        return storage_url

    raw_path = storage_url.removeprefix("sqlite:///")
    if raw_path == ":memory:":
        return storage_url

    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = resolve_path(raw_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _build_optuna_sampler(optuna: Any, tuning_cfg: dict[str, Any]) -> Any:
    """Create a reproducible Optuna sampler from config."""
    sampler_name = str(tuning_cfg.get("sampler", "tpe")).strip().lower()
    seed = tuning_cfg.get("seed")
    n_startup_trials = int(tuning_cfg.get("n_startup_trials", 10))

    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler '{sampler_name}'. Choose from ['random', 'tpe'].")


def _build_optuna_pruner(optuna: Any, tuning_cfg: dict[str, Any]) -> Any:
    """Create a pruner that can safely stop weak trials early."""
    pruner_name = str(tuning_cfg.get("pruner", "median")).strip().lower()
    n_startup_trials = int(tuning_cfg.get("n_startup_trials", 10))
    n_warmup_steps = int(tuning_cfg.get("n_warmup_steps", 1))

    if pruner_name in {"none", "nop", "disabled"}:
        return optuna.pruners.NopPruner()
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(
        "Unsupported Optuna pruner "
        f"'{pruner_name}'. Choose from ['disabled', 'hyperband', 'median', 'successive_halving']."
    )


def _apply_smote_for_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    smote_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE for a tuning fold and fall back cleanly when a fold is too small."""
    try:
        return apply_smote(X_train, y_train, smote_cfg=smote_cfg)
    except ValueError as exc:
        if smote_cfg.get("enabled", False):
            logger.warning("Skipping SMOTE for this tuning fold: %s", exc)
        return X_train, np.asarray(y_train)


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    n_trials: int | None = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any], dict[str, Any]]:
    """Tune XGBoost with Optuna on chronological folds."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for tune_xgboost.") from exc

    model_cfg = load_config("model")
    paths_cfg = load_config("paths")
    tuning_cfg = model_cfg.get("tuning", {})
    smote_cfg = dict(model_cfg.get("smote", {}))
    xgb_params = dict(model_cfg.get("xgboost", {}))

    if n_trials is None:
        n_trials = int(tuning_cfg.get("n_trials", 25))

    n_splits = int(tuning_cfg.get("n_splits", 4))
    timeout = tuning_cfg.get("timeout_seconds")
    metric_name = tuning_cfg.get("metric", "average_precision")
    study_name = str(tuning_cfg.get("study_name", "xgboost_tuning")).strip() or "xgboost_tuning"
    experiments_dir = resolve_path(paths_cfg["models"]["experiments_dir"])
    storage_url = _resolve_storage_url(
        tuning_cfg.get("storage"),
        study_name=study_name,
        experiments_dir=experiments_dir,
    )
    sampler = _build_optuna_sampler(optuna, tuning_cfg)
    pruner = _build_optuna_pruner(optuna, tuning_cfg)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.5),
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores: list[float] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X_train), start=1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_valid = y_train[valid_idx]
            X_fold_train_fit, y_fold_train_fit = _apply_smote_for_tuning(
                X_fold_train,
                y_fold_train,
                smote_cfg=smote_cfg,
            )

            fold_params, early_stopping = _prepare_xgboost_params(
                y_fold_train_fit,
                params_override=trial_params,
                xgb_cfg=xgb_params,
            )
            if early_stopping is not None:
                fold_params["early_stopping_rounds"] = early_stopping

            model = xgb.XGBClassifier(**fold_params)
            model.fit(
                X_fold_train_fit,
                y_fold_train_fit,
                eval_set=[(X_fold_valid, y_fold_valid)],
                verbose=False,
            )
            y_prob = model.predict_proba(X_fold_valid)[:, 1]
            fold_score = _score_predictions(metric_name, y_fold_valid, y_prob)
            trial.report(float(fold_score), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            scores.append(float(fold_score))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=bool(tuning_cfg.get("load_if_exists", True)),
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = dict(study.best_params)
    logger.info("Best Optuna %s: %.4f", metric_name, study.best_value)
    logger.info("Best Optuna params: %s", best_params)
    X_train_fit, y_train_fit = apply_smote(X_train, y_train, smote_cfg=smote_cfg)
    final_model = train_xgboost(
        X_train_fit,
        y_train_fit,
        X_valid,
        y_valid,
        params_override=best_params,
        fit_verbose=False,
    )

    study_summary = {
        "metric": metric_name,
        "best_value": float(study.best_value),
        "study_name": study.study_name,
        "storage": storage_url,
        "sampler": str(tuning_cfg.get("sampler", "tpe")).strip().lower(),
        "pruner": str(tuning_cfg.get("pruner", "median")).strip().lower(),
    }
    return final_model, best_params, study_summary
