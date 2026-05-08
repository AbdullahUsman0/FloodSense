from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from floodsense.config import (
    DISTRICT_RISK_PROFILE,
    DISTRICT_TO_REGION,
    HIGH_THRESHOLD,
    INSUFFICIENT_DATA_MESSAGE,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    REGION_DISTRICT_DIVISOR,
    RISK_ACTIONS,
    RISK_COLORS,
    RISK_LABELS_UR,
    SOIL_CONDITION_TO_VALUE,
    VISIBLE_WATER_OPTIONS,
    RiskResult,
)

_RISK_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def risk_from_probability(probability: float) -> tuple[str, str]:
    if probability < LOW_THRESHOLD:
        return "Low", RISK_LABELS_UR["Low"]
    if probability <= MEDIUM_THRESHOLD:
        return "Medium", RISK_LABELS_UR["Medium"]
    if probability <= HIGH_THRESHOLD:
        return "High", RISK_LABELS_UR["High"]
    return "Critical", RISK_LABELS_UR["Critical"]


def risk_color(risk_label: str) -> str:
    return RISK_COLORS[risk_label]


def estimate_population_at_risk(
    district: str,
    ndma_population_by_region: dict[str, int],
    risk_level: str,
    confidence_pct: float,
) -> int:
    region = DISTRICT_TO_REGION.get(district)
    if region is None or region not in ndma_population_by_region:
        return 0
    regional_population = ndma_population_by_region[region]
    divisor = REGION_DISTRICT_DIVISOR.get(region, 1)
    district_baseline = regional_population / max(divisor, 1)
    district_profile = DISTRICT_RISK_PROFILE.get(
        district,
        {"settlement_type": "default", "exposure_factor": 0.75, "max_exposed_share": 0.06},
    )
    exposure_factor = float(district_profile["exposure_factor"])
    profile_cap_share = float(district_profile["max_exposed_share"])

    severity_rate = {
        "Low": 0.002,
        "Medium": 0.007,
        "High": 0.015,
        "Critical": 0.030,
    }[risk_level]
    max_cap_rate = {
        "Low": 0.01,
        "Medium": 0.03,
        "High": 0.06,
        "Critical": 0.10,
    }[risk_level]
    confidence_factor = 0.6 + (0.4 * (confidence_pct / 100.0))
    raw_estimate = district_baseline * severity_rate * confidence_factor * exposure_factor
    cap_value = district_baseline * min(max_cap_rate, profile_cap_share)
    bounded_estimate = min(raw_estimate, cap_value)
    minimum_by_risk = {
        "Low": 50,
        "Medium": 150,
        "High": 300,
        "Critical": 500,
    }[risk_level]
    bounded_estimate = max(minimum_by_risk, bounded_estimate)
    return int(round(bounded_estimate, -2))


def apply_operational_risk_guardrail(
    base_risk_level_en: str,
    rainfall_mm: float,
    soil_condition: str,
    visible_water: str,
) -> str:
    """
    Operational safety override:
    - Extreme rainfall + saturated/water-visible conditions should not under-warn.
    """
    forced_level = base_risk_level_en
    if rainfall_mm >= 450 and soil_condition == "Saturated" and visible_water == "Yes":
        forced_level = "Critical"
    elif rainfall_mm >= 350 and (soil_condition == "Saturated" or visible_water == "Yes"):
        forced_level = "High"
    elif rainfall_mm >= 300 and soil_condition in {"Moist", "Saturated"} and visible_water == "Yes":
        forced_level = "High"

    if _RISK_ORDER[forced_level] > _RISK_ORDER[base_risk_level_en]:
        return forced_level
    return base_risk_level_en


def validate_user_inputs(
    rainfall_mm: float | None,
    selected_date: date | None,
    district: str | None,
    soil_condition: str | None,
    visible_water: str | None,
    allowed_districts: set[str],
) -> tuple[bool, str]:
    if rainfall_mm is None or selected_date is None or not district or not soil_condition or not visible_water:
        return False, INSUFFICIENT_DATA_MESSAGE
    if rainfall_mm < 0 or rainfall_mm > 500:
        return False, INSUFFICIENT_DATA_MESSAGE
    if district not in allowed_districts:
        return False, INSUFFICIENT_DATA_MESSAGE
    if soil_condition not in SOIL_CONDITION_TO_VALUE:
        return False, INSUFFICIENT_DATA_MESSAGE
    if visible_water not in VISIBLE_WATER_OPTIONS:
        return False, INSUFFICIENT_DATA_MESSAGE
    return True, ""


def build_input_row(
    rainfall_mm: float,
    selected_date: date,
    district: str,
    soil_condition: str,
    visible_water: str,
    baseline_payload: dict[str, Any],
) -> pd.DataFrame:
    month = selected_date.month
    day_of_year = selected_date.timetuple().tm_yday
    key = f"{district}::{month}"

    base = baseline_payload["district_month"].get(key, baseline_payload["default"]).copy()
    terrain = baseline_payload["terrain_lookup"].get(
        district,
        {"terrain_type": "Unknown", "avg_elevation_m": float(base.get("avg_elevation_m", np.nan))},
    )

    row: dict[str, Any] = dict(base)
    row["district"] = district
    row["terrain_type"] = terrain["terrain_type"]
    row["avg_elevation_m"] = float(terrain["avg_elevation_m"])

    row["month"] = month
    row["day_of_year"] = day_of_year
    row["year"] = selected_date.year
    row["is_monsoon"] = int(month in [7, 8, 9])

    row["precipitation"] = float(rainfall_mm)
    row["precip_3day_avg"] = float(0.5 * rainfall_mm + 0.5 * float(base.get("precip_3day_avg", rainfall_mm)))
    row["precip_7day_avg"] = float(0.3 * rainfall_mm + 0.7 * float(base.get("precip_7day_avg", rainfall_mm)))

    soil_value = SOIL_CONDITION_TO_VALUE[soil_condition]
    row["soil_moisture"] = soil_value
    row["soil_3day_avg"] = float(0.6 * float(base.get("soil_3day_avg", soil_value)) + 0.4 * soil_value)

    water_base = float(base.get("water_area_km2", 0.0))
    if visible_water == "Yes":
        row["water_area_km2"] = max(water_base, (1.2 * water_base) + 5.0)
        row["water_area_change"] = max(1.0, abs(float(base.get("water_area_change", 0.0))) + 2.5)
    else:
        row["water_area_km2"] = max(0.0, 0.8 * water_base)
        row["water_area_change"] = min(0.0, float(base.get("water_area_change", 0.0)))

    prev_water = max(row["water_area_km2"] - row["water_area_change"], 0.001)
    row["water_area_pct_change"] = float(row["water_area_change"] / prev_water)
    row["rain_soil_interaction"] = float(row["precipitation"] * row["soil_moisture"])
    prior_cumulative = float(base.get("monsoon_cumulative_precip", 0.0))
    row["monsoon_cumulative_precip"] = float(prior_cumulative + row["precipitation"] if row["is_monsoon"] == 1 else 0.0)
    row["water_area_acceleration"] = float(row["water_area_change"] - float(base.get("water_area_change", 0.0)))

    return pd.DataFrame([row])


def predict_risk(
    model: Any,
    input_row: pd.DataFrame,
    district: str,
    ndma_population_by_region: dict[str, int],
) -> RiskResult:
    flood_probability = float(model.predict_proba(input_row)[0][1])
    confidence_pct = float(max(flood_probability, 1.0 - flood_probability) * 100.0)
    risk_en, risk_ur = risk_from_probability(flood_probability)
    action = RISK_ACTIONS[risk_en]
    estimated_pop = estimate_population_at_risk(district, ndma_population_by_region, risk_en, confidence_pct)
    return RiskResult(
        risk_level_en=risk_en,
        risk_level_ur=risk_ur,
        confidence_pct=round(confidence_pct, 1),
        recommended_action_en=action["en"],
        recommended_action_ur=action["ur"],
        population_risk_estimate=estimated_pop,
    )


def format_result_payload(result: RiskResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["risk_color"] = risk_color(result.risk_level_en)
    return payload


def load_artifacts(artifacts_dir: Path) -> tuple[Any, dict[str, Any]]:
    model = joblib.load(artifacts_dir / "model.joblib")
    with (artifacts_dir / "model_meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return model, meta


def predict_from_user_inputs(
    model: Any,
    meta: dict[str, Any],
    rainfall_mm: float,
    selected_date: date,
    district: str,
    soil_condition: str,
    visible_water: str,
) -> dict[str, Any]:
    allowed_districts = set(meta.get("allowed_districts", []))
    ok, message = validate_user_inputs(
        rainfall_mm=rainfall_mm,
        selected_date=selected_date,
        district=district,
        soil_condition=soil_condition,
        visible_water=visible_water,
        allowed_districts=allowed_districts,
    )
    if not ok:
        return {
            "ok": False,
            "message": message,
        }

    row = build_input_row(
        rainfall_mm=rainfall_mm,
        selected_date=selected_date,
        district=district,
        soil_condition=soil_condition,
        visible_water=visible_water,
        baseline_payload=meta["baseline_payload"],
    )
    result = predict_risk(
        model=model,
        input_row=row,
        district=district,
        ndma_population_by_region=meta["ndma_population_by_region"],
    )
    adjusted_level = apply_operational_risk_guardrail(
        base_risk_level_en=result.risk_level_en,
        rainfall_mm=rainfall_mm,
        soil_condition=soil_condition,
        visible_water=visible_water,
    )
    if adjusted_level != result.risk_level_en:
        # Keep confidence from model but lift risk class for operational safety.
        result = RiskResult(
            risk_level_en=adjusted_level,
            risk_level_ur=RISK_LABELS_UR[adjusted_level],
            confidence_pct=result.confidence_pct,
            recommended_action_en=RISK_ACTIONS[adjusted_level]["en"],
            recommended_action_ur=RISK_ACTIONS[adjusted_level]["ur"],
            population_risk_estimate=estimate_population_at_risk(
                district=district,
                ndma_population_by_region=meta["ndma_population_by_region"],
                risk_level=adjusted_level,
                confidence_pct=result.confidence_pct,
            ),
        )
    payload = format_result_payload(result)
    payload["ok"] = True
    return payload
