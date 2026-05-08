from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from floodsense.config import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    report: dict[str, Any]


def build_preprocessor() -> ColumnTransformer:
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
            ("num", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
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
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return EvaluationResult(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=cm,
        report=report,
    )


def choose_best_model(scorecard: dict[str, dict[str, EvaluationResult]]) -> str:
    best_name = ""
    best_score = -np.inf
    for model_name, splits in scorecard.items():
        strat = splits["stratified"]
        time_based = splits["time_based"]
        score = (0.4 * strat.accuracy) + (0.6 * time_based.accuracy) + (0.2 * time_based.f1)
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
