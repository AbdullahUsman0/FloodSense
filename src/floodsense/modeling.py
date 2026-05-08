from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from floodsense.config import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    report: dict[str, Any]


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = MODEL_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = MODEL_CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_candidate_pipelines(
    random_state: int = 42,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }
    return {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
        for name, clf in models.items()
    }


def evaluate_binary_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return EvaluationResult(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        confusion_matrix=cm,
        report=report,
    )


def choose_best_model(scorecard: dict[str, dict[str, EvaluationResult]]) -> str:
    best_name = ""
    best_score = -np.inf
    for model_name, splits in scorecard.items():
        year_holdout = splits["year_holdout"]
        timeseries_cv = splits["timeseries_cv"]
        score = (
            (0.60 * year_holdout.recall)
            + (0.30 * year_holdout.roc_auc)
            + (0.20 * timeseries_cv.recall)
            + (0.10 * timeseries_cv.roc_auc)
        )
        if score > best_score:
            best_score = score
            best_name = model_name
    if not best_name:
        raise ValueError("No model candidates available.")
    return best_name


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    feature_names = preprocessor.get_feature_names_out()
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df
