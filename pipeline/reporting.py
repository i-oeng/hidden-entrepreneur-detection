# reporting.py - Reproducibility metadata export

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_run_metadata(
    out_dir: Path,
    config_module,
    best_params: dict,
    val_results: dict,
    card_count: int,
    consumer_count: int,
    business_count: int,
) -> None:
    """Persist the main knobs and learned validation artifacts for reruns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": config_module.DATA_DIR,
        "out_dir": config_module.OUT_DIR,
        "seed": config_module.SEED,
        "n_bags": config_module.N_BAGS,
        "bag_ratio": config_module.BAG_RATIO,
        "reliable_neg_quantile": config_module.RELIABLE_NEG_QUANTILE,
        "n_optuna_trials": config_module.N_OPTUNA_TRIALS,
        "n_cv_folds": config_module.N_CV_FOLDS,
        "default_ensemble_weights": config_module.ENSEMBLE_WEIGHTS,
        "tuned_ensemble_weights": val_results.get("ensemble_weights"),
        "best_validation_threshold": val_results.get("best_thresh"),
        "best_validation_fbeta": val_results.get("best_fbeta"),
        "best_lgb_params": best_params,
        "feature_count": len(config_module.FEATURES),
        "features": config_module.FEATURES,
        "card_count": card_count,
        "consumer_count": consumer_count,
        "business_count": business_count,
    }
    path = out_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2, ensure_ascii=False)
    print(f"Run metadata saved to {path}")
