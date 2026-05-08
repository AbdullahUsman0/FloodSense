from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.config import MODEL_NUMERIC_FEATURES
from floodsense.data_pipeline import (
    attach_elevation_features,
    clean_training_data,
    get_feature_target_frame,
    load_raw_data,
    time_based_split,
)
from floodsense.modeling import build_candidate_pipelines
from floodsense.scenario import apply_scenario_card1_monsoon_surge, apply_scenario_card2_sensor_rogue


def metric_block(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_flood": precision_score(y_test, y_pred, zero_division=0),
        "recall_flood": recall_score(y_test, y_pred, zero_division=0),
        "f1_flood": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_eval(feature_drop: list[str]) -> dict[str, object]:
    train_df, elevation_df, _ = load_raw_data(ROOT)
    train_df, _ = apply_scenario_card2_sensor_rogue(train_df, faulty_district="Sindh_District")
    train_df, _ = apply_scenario_card1_monsoon_surge(train_df)
    cleaned, _ = clean_training_data(train_df)
    model_df = attach_elevation_features(cleaned, elevation_df)
    x_all, y = get_feature_target_frame(model_df)
    x_all = x_all.drop(columns=[c for c in feature_drop if c in x_all.columns])

    pipelines = build_candidate_pipelines()
    rf = clone(pipelines["random_forest"])
    kept_numeric = [c for c in MODEL_NUMERIC_FEATURES if c in x_all.columns]
    rf.named_steps["preprocessor"].transformers[0] = ("num", rf.named_steps["preprocessor"].transformers[0][1], kept_numeric)

    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
        x_all, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train_t, x_test_t, y_train_t, y_test_t = time_based_split(x_all, y, cleaned["date"], test_fraction=0.2)

    strat_model = clone(rf).fit(x_train_s, y_train_s)
    time_model = clone(rf).fit(x_train_t, y_train_t)
    return {
        "stratified": metric_block(strat_model, x_test_s, y_test_s),
        "time_based": metric_block(time_model, x_test_t, y_test_t),
    }


def main() -> None:
    train_df, elevation_df, _ = load_raw_data(ROOT)
    train_df, _ = apply_scenario_card2_sensor_rogue(train_df, faulty_district="Sindh_District")
    train_df, _ = apply_scenario_card1_monsoon_surge(train_df)
    cleaned, stats = clean_training_data(train_df)
    model_df = attach_elevation_features(cleaned, elevation_df)

    numeric = model_df.select_dtypes(include=[np.number]).copy()
    correlations = (
        numeric.corr(numeric_only=True)["flood_event"]
        .drop(labels=["flood_event"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(12)
    )

    exact_duplicate_keys = cleaned.duplicated(subset=["district", "date"]).sum()
    flood_rate = float(model_df["flood_event"].mean())

    full = train_eval([])
    no_water_extent = train_eval(
        ["water_area_km2", "water_area_change", "water_area_pct_change", "water_area_acceleration"]
    )
    no_temporal = train_eval(["month", "day_of_year", "is_monsoon", "monsoon_cumulative_precip"])

    report = {
        "data_quality": {
            "rows_before_cleanup": stats.rows_before,
            "rows_after_cleanup": stats.rows_after,
            "duplicates_removed": stats.duplicates_removed,
            "duplicate_district_date_after_cleanup": int(exact_duplicate_keys),
            "phantom_rows_removed": stats.phantom_rows_removed,
            "missing_precipitation_handled": stats.precipitation_missing_count,
            "non_finite_water_pct_handled": stats.water_pct_non_finite_count,
            "flood_event_rate": flood_rate,
        },
        "top_abs_target_correlations": correlations.to_dict(),
        "full_feature_model": full,
        "without_water_extent_features": no_water_extent,
        "without_temporal_features": no_temporal,
        "leakage_assessment": {
            "direct_target_column_in_features": False,
            "known_forbidden_training_columns_used": False,
            "risk_level": "medium",
            "reason": (
                "No direct target leakage was found, but water extent features almost perfectly separate flood "
                "events and dominate model importance. This is plausible for satellite flood monitoring, yet it "
                "should be defended as an observed hazard signal, not as a future-only label proxy."
            ),
        },
    }
    out = ROOT / "artifacts" / "leakage_and_metrics_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
