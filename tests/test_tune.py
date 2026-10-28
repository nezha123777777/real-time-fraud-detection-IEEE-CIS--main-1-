from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.tune import _resolve_storage_url, _score_predictions, tune_xgboost


class TestScorePredictions:
    def test_supports_metric_aliases(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])

        assert _score_predictions("average_precision", y_true, y_prob) == pytest.approx(1.0)
        assert _score_predictions("pr_auc", y_true, y_prob) == pytest.approx(1.0)
        assert _score_predictions("aucpr", y_true, y_prob) == pytest.approx(1.0)
        assert _score_predictions("roc_auc", y_true, y_prob) == pytest.approx(1.0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unsupported Optuna metric"):
            _score_predictions("f1", np.array([0, 1]), np.array([0.2, 0.8]))


class TestResolveStorageUrl:
    def test_defaults_to_persistent_sqlite_in_experiments_dir(self, tmp_path):
        storage = _resolve_storage_url(None, study_name="fraud-study", experiments_dir=tmp_path)

        assert storage == f"sqlite:///{(tmp_path / 'fraud-study.db').resolve().as_posix()}"
        assert tmp_path.exists()

    def test_resolves_relative_sqlite_paths(self, tmp_path, monkeypatch):
        experiments_dir = tmp_path / "artifacts" / "models" / "experiments"
        experiments_dir.mkdir(parents=True)
        relative_storage = "sqlite:///artifacts/models/experiments/optuna.db"
        monkeypatch.setattr(
            "src.models.tune.resolve_path",
            lambda relative: (tmp_path / relative).resolve(),
        )

        storage = _resolve_storage_url(
            relative_storage,
            study_name="fraud-study",
            experiments_dir=experiments_dir,
        )

        assert storage == f"sqlite:///{(tmp_path / 'artifacts/models/experiments/optuna.db').resolve().as_posix()}"


class TestTuneXGBoost:
    def test_uses_persistent_study_and_final_training_flow(self, monkeypatch, tmp_path):
        class FakeTrial:
            def __init__(self):
                self.params: dict[str, float | int] = {}

            def suggest_int(self, name, low, high):
                value = low
                self.params[name] = value
                return value

            def suggest_float(self, name, low, high, log=False):
                value = low
                self.params[name] = value
                return value

            def report(self, value, step):
                self.last_report = (value, step)

            def should_prune(self):
                return False

        class FakeStudy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.study_name = kwargs["study_name"]
                self.best_params: dict[str, float | int] = {}
                self.best_value = 0.0

            def optimize(self, objective, n_trials, timeout):
                trial = FakeTrial()
                self.best_value = objective(trial)
                self.best_params = dict(trial.params)

        class FakeTPESampler:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeMedianPruner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        create_study_calls: list[FakeStudy] = []

        def create_study(**kwargs):
            study = FakeStudy(**kwargs)
            create_study_calls.append(study)
            return study

        fake_optuna = SimpleNamespace(
            Trial=FakeTrial,
            TrialPruned=type("TrialPruned", (Exception,), {}),
            samplers=SimpleNamespace(
                TPESampler=FakeTPESampler,
                RandomSampler=FakeTPESampler,
            ),
            pruners=SimpleNamespace(
                MedianPruner=FakeMedianPruner,
                NopPruner=FakeMedianPruner,
                SuccessiveHalvingPruner=FakeMedianPruner,
                HyperbandPruner=FakeMedianPruner,
            ),
            create_study=create_study,
        )

        monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

        tune_storage = tmp_path / "optuna.db"
        model_cfg = {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "n_estimators": 20,
                "early_stopping_rounds": 5,
                "scale_pos_weight": None,
                "random_state": 42,
                "n_jobs": 1,
            },
            "smote": {
                "enabled": True,
                "sampling_strategy": "auto",
                "k_neighbors": 2,
                "random_state": 42,
            },
            "tuning": {
                "metric": "pr_auc",
                "n_trials": 1,
                "n_splits": 2,
                "timeout_seconds": 30,
                "study_name": "fraud-study",
                "storage": f"sqlite:///{tune_storage.as_posix()}",
                "load_if_exists": True,
                "seed": 123,
                "sampler": "tpe",
                "pruner": "median",
                "n_startup_trials": 4,
                "n_warmup_steps": 1,
            },
        }
        paths_cfg = {
            "models": {
                "experiments_dir": "artifacts/models/experiments",
            }
        }

        def fake_load_config(name: str):
            if name == "model":
                return model_cfg
            if name == "paths":
                return paths_cfg
            raise AssertionError(f"Unexpected config requested: {name}")

        smote_calls: list[tuple[int, int]] = []

        def fake_apply_smote(X_train, y_train, smote_cfg=None):
            smote_calls.append((len(X_train), len(y_train)))
            return X_train, np.asarray(y_train)

        class DummyXGBClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, X_train, y_train, eval_set=None, verbose=False):
                self.fit_shape = (len(X_train), len(y_train))
                self.eval_set = eval_set
                self.verbose = verbose
                return self

            def predict_proba(self, X_valid):
                scores = np.clip(np.asarray(X_valid["score"]), 1e-6, 1 - 1e-6)
                return np.column_stack([1.0 - scores, scores])

        final_train_calls: list[dict[str, object]] = []

        def fake_train_xgboost(
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            params_override=None,
            fit_verbose=100,
        ):
            final_train_calls.append({
                "train_rows": len(X_train),
                "valid_rows": len(X_valid) if X_valid is not None else None,
                "params_override": params_override,
                "fit_verbose": fit_verbose,
            })
            return "trained-model"

        monkeypatch.setattr("src.models.tune.load_config", fake_load_config)
        monkeypatch.setattr("src.models.tune.apply_smote", fake_apply_smote)
        monkeypatch.setattr("src.models.tune.train_xgboost", fake_train_xgboost)
        monkeypatch.setattr("src.models.tune.xgb.XGBClassifier", DummyXGBClassifier)

        X_train = pd.DataFrame({
            "score": [0.05, 0.95, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        })
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        X_valid = pd.DataFrame({"score": [0.15, 0.85]})
        y_valid = np.array([0, 1])

        model, best_params, study_summary = tune_xgboost(X_train, y_train, X_valid, y_valid, n_trials=1)

        assert model == "trained-model"
        assert best_params["max_depth"] == 3
        assert study_summary["study_name"] == "fraud-study"
        assert study_summary["storage"] == f"sqlite:///{tune_storage.resolve().as_posix()}"
        assert len(smote_calls) == 3
        assert final_train_calls[0]["params_override"] == best_params
        assert final_train_calls[0]["fit_verbose"] is False

        study = create_study_calls[0]
        assert study.kwargs["study_name"] == "fraud-study"
        assert study.kwargs["storage"] == f"sqlite:///{tune_storage.resolve().as_posix()}"
        assert study.kwargs["load_if_exists"] is True
        assert study.kwargs["sampler"].kwargs["seed"] == 123
