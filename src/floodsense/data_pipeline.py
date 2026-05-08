from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from floodsense.config import (
    DROP_COLUMNS,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    OUTLIER_BOUNDS,
)


@dataclass
class DataQualityStats:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    phantom_rows_removed: int
    bad_date_rows_removed: int
    precipitation_missing_count: int
    water_pct_non_finite_count: int


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "floodsense_training_data.csv")
    elevation_df = pd.read_csv(data_dir / "district_elevation_reference.csv")
    ndma_df = pd.read_csv(data_dir / "ndma_flood_impact_2022.csv")
    return train_df, elevation_df, ndma_df


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _is_phantom_row(row: pd.Series) -> bool:
    for col, (low, high) in OUTLIER_BOUNDS.items():
        value = row.get(col)
        if pd.isna(value):
            continue
        if low is not None and value < low:
            return True
        if high is not None and value > high:
            return True
    return False


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityStats]:
    working = df.copy()
    rows_before = len(working)

    working["date"] = pd.to_datetime(working["date"], format="%m/%d/%Y", errors="coerce")
    bad_date_rows = working["date"].isna().sum()
    working = working.dropna(subset=["date"])

    numeric_columns = [
        "elevation",
        "evaporation",
        "latitude",
        "longitude",
        "precipitation",
        "pressure",
        "soil_moisture",
        "temperature",
        "water_area_km2",
        "wind_speed",
        "humidity",
        "precip_3day_avg",
        "precip_7day_avg",
        "temp_3day_avg",
        "soil_3day_avg",
        "day_of_year",
        "month",
        "year",
        "is_monsoon",
        "water_area_change",
        "water_area_pct_change",
        "ds_idx",
        "flood_event",
    ]
    for col in numeric_columns:
        working[col] = _coerce_numeric(working, col)

    # Sentinel cleanup required by challenge brief.
    working.loc[working["precipitation"] == -999, "precipitation"] = np.nan
    precipitation_missing_count = working["precipitation"].isna().sum()

    working["water_area_pct_change"] = working["water_area_pct_change"].replace([np.inf, -np.inf], np.nan)
    water_pct_non_finite_count = working["water_area_pct_change"].isna().sum()

    phantom_mask = working.apply(_is_phantom_row, axis=1)
    phantom_rows_removed = int(phantom_mask.sum())
    working = working.loc[~phantom_mask].copy()

    before_dedup = len(working)
    working = working.drop_duplicates().copy()
    duplicates_removed = before_dedup - len(working)

    # Recompute time helpers from parsed date for consistency.
    working["month"] = working["date"].dt.month
    working["day_of_year"] = working["date"].dt.dayofyear
    working["year"] = working["date"].dt.year
    working["is_monsoon"] = working["month"].isin([7, 8, 9]).astype(int)
    working = working.sort_values(["district", "date"]).reset_index(drop=True)

    # Winsorize water area percentage change after cleaning.
    low = working["water_area_pct_change"].quantile(0.01)
    high = working["water_area_pct_change"].quantile(0.99)
    working["water_area_pct_change"] = working["water_area_pct_change"].clip(lower=low, upper=high)

    # Feature engineering aligned with hackathon guidance.
    working["rain_soil_interaction"] = working["precipitation"].fillna(0.0) * working["soil_moisture"].fillna(0.0)
    working["monsoon_precip_raw"] = np.where(working["is_monsoon"] == 1, working["precipitation"].fillna(0.0), 0.0)
    working["monsoon_cumulative_precip"] = working.groupby(["district", "year"], sort=False)["monsoon_precip_raw"].cumsum()
    working["water_area_km2_lag1"] = working.groupby("district", sort=False)["water_area_km2"].shift(1)
    working["water_area_change_lag1"] = working.groupby("district", sort=False)["water_area_change"].shift(1)
    working["water_area_pct_change_lag1"] = working.groupby("district", sort=False)["water_area_pct_change"].shift(1)
    for lag_col in ["water_area_km2_lag1", "water_area_change_lag1", "water_area_pct_change_lag1"]:
        working[lag_col] = working[lag_col].fillna(working[lag_col].median())
    working = working.drop(columns=["monsoon_precip_raw"], errors="ignore")
    working = working.sort_values("date").reset_index(drop=True)

    stats = DataQualityStats(
        rows_before=rows_before,
        rows_after=len(working),
        duplicates_removed=duplicates_removed,
        phantom_rows_removed=phantom_rows_removed,
        bad_date_rows_removed=int(bad_date_rows),
        precipitation_missing_count=int(precipitation_missing_count),
        water_pct_non_finite_count=int(water_pct_non_finite_count),
    )
    return working, stats


def attach_elevation_features(train_df: pd.DataFrame, elevation_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_df.merge(elevation_df, on="district", how="left", validate="many_to_one")
    return merged


def get_feature_target_frame(df: pd.DataFrame, drop_feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    y = df["flood_event"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
    if drop_feature_columns:
        X = X.drop(columns=[c for c in drop_feature_columns if c in X.columns], errors="ignore")
    return X, y


def assert_no_non_finite_inputs(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Found infinite values in numeric training inputs.")


def build_district_month_baseline(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    def _to_native(value: Any) -> Any:
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    baseline_cols = MODEL_NUMERIC_FEATURES + ["year"]
    grouped = df.groupby(["district", "month"], as_index=False)[baseline_cols].median(numeric_only=True)
    default_group = {k: _to_native(v) for k, v in df[baseline_cols].median(numeric_only=True).to_dict().items()}
    terrain_map = df.drop_duplicates(subset=["district"])[["district", "terrain_type", "avg_elevation_m"]]

    baseline: dict[str, dict[str, Any]] = {}
    for _, row in grouped.iterrows():
        key = f"{row['district']}::{int(row['month'])}"
        baseline[key] = {k: _to_native(v) for k, v in row.to_dict().items()}

    terrain_lookup = {
        r["district"]: {"terrain_type": r["terrain_type"], "avg_elevation_m": float(r["avg_elevation_m"])}
        for _, r in terrain_map.iterrows()
    }
    return {
        "district_month": baseline,
        "default": default_group,
        "terrain_lookup": terrain_lookup,
    }


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sort_idx = np.argsort(dates.to_numpy())
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)

    split_idx = int((1.0 - test_fraction) * len(X_sorted))
    X_train = X_sorted.iloc[:split_idx].copy()
    X_test = X_sorted.iloc[split_idx:].copy()
    y_train = y_sorted.iloc[:split_idx].copy()
    y_test = y_sorted.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test
