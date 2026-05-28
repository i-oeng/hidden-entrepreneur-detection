# save_models.py - Persist trained models for later scoring

import pickle
from pathlib import Path

import numpy as np


def save_models(
    lgb_model,
    catboost_model,
    iso_model,
    iso_ref: np.ndarray,
    score_calibrator,
    ensemble_weights: dict,
    out_dir: Path,
) -> None:
    """Save all trained model artifacts to out_dir/models/."""
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "lgb_model"      : lgb_model,
        "catboost_model"  : catboost_model,
        "iso_model"       : iso_model,
        "iso_ref"         : iso_ref,
        "score_calibrator": score_calibrator,
        "ensemble_weights": ensemble_weights,
    }

    for name, obj in artifacts.items():
        path = model_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    print(f"\nAll models saved to {model_dir}/")
    for name in artifacts:
        print(f"  {name}.pkl")
