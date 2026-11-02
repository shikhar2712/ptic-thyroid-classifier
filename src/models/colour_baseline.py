"""Shortcut baselines: image-level statistics -> logistic regression. None of them looks at nuclear morphology.

B0 = colour/contrast statistics, B1 = one texture scalar (Laplacian variance, i.e. fine-structure energy),
B2 = both. Whatever they achieve on patient-grouped folds is achievable without recognising PTC nuclear features, so a CNN
must be compared against them before any claim that it "learned PTC morphology".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "gray_std", "white_frac"]  # B0
TEXTURE_FEATURES = ["lap_var"]                                                                    # B1
BASELINES = {"B0_colour": FEATURES, "B1_texture": TEXTURE_FEATURES, "B2_colour_texture": FEATURES + TEXTURE_FEATURES}


def colour_features(manifest: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    return manifest[features or FEATURES].to_numpy(dtype=np.float64)


def fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame, C: float = 1.0, seed: int = 0,
                features: list[str] | None = None) -> np.ndarray:
    """Fit on `train` rows only (scaler included) and return P(PTC-like) for `evaluate` rows."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
    clf.fit(colour_features(train, features), train.label.to_numpy())
    return clf.predict_proba(colour_features(evaluate, features))[:, 1]
