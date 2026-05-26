# tuning.py - Hyperparameter Tuning + Model Training
# Tune LightGBM via Optuna and train final ensemble.

import numpy as np
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def tune_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pos: np.ndarray,
    pos_weight: float,
    seed: int,
    n_trials: int,
    n_cv_folds: int,
    catboost_params: dict,
    iso_params: dict,
):
    """Run Optuna hyperparameter search for LightGBM, then train all three models."""
    print("\n" + "=" * 60)
    print(f"SECTION 6: LightGBM Hyperparameter Tuning ({n_trials} Optuna trials)")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    def lgb_objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves"       : trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            "scale_pos_weight" : pos_weight,
            "n_jobs"           : -1,
            "verbose"          : -1,
            "random_state"     : seed,
        }
        fold_scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            Xtr, Xva = X_train[tr_idx], X_train[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            fold_scores.append(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": pos_weight,
        "n_jobs"          : -1,
        "verbose"         : -1,
        "random_state"    : seed,
    })
    print(f"\nBest CV PR-AUC : {study.best_value:.4f}")
    print(f"Best params    : {best_params}")

    # Train final LightGBM on full clean training set
    lgb_model = lgb.LGBMClassifier(**best_params)
    lgb_model.fit(X_train, y_train)

    # CatBoost
    print("\nTraining CatBoost...")
    cb_params = dict(catboost_params)
    cb_params["class_weights"] = {0: 1, 1: pos_weight}
    catboost_model = CatBoostClassifier(**cb_params)
    catboost_model.fit(X_train, y_train)

    # IsolationForest (unsupervised, positives only)
    print("Training Isolation Forest (unsupervised, positives-only)...")
    iso_model = IsolationForest(**iso_params)
    iso_model.fit(X_pos)

    return lgb_model, catboost_model, iso_model, best_params
