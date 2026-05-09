"""
run_prediction.py — FloodSense Prediction Leaderboard
Neural Nova · BTech Fest 2026 · BNU Lahore
------------------------------------------------------
DROP THIS FILE into your project folder (same level as your trained model).
Run: python run_prediction.py
Your submission file will be saved as: floodsense_submission.csv

Requirements:
- Your trained model must be loadable (joblib or pickle).
- Update the THREE configuration values in the USER CONFIG section below.
- Do not modify anything outside the USER CONFIG section.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG — edit these three values only
# ─────────────────────────────────────────────────────────────────────────────

TEAM_NAME = "Team AU"           # e.g. "Team Nexus"
INSTITUTION = "FAST NUCES, Lahore"       # e.g. "FAST NUCES, Lahore"

# Path to your saved model file — supports joblib (.pkl, .joblib) or pickle
MODEL_PATH = "artifacts/model.joblib"               # e.g. "model/random_forest.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "district": "Nowshera",
        "precipitation": 142.5, "precip_3day_avg": 98.2, "precip_7day_avg": 61.4,
        "soil_moisture": 0.74, "soil_3day_avg": 0.68,
        "water_area_km2": 890.3, "water_area_change": 210.5, "water_area_pct_change": 0.31,
        "temperature": 31.2, "humidity": 88.0, "pressure": 98800.0,
        "evaporation": -0.0018, "wind_speed": 4.1,
        "month": 8, "day_of_year": 227, "is_monsoon": 1, "ds_idx": 3.0,
    },
    {
        "district": "Jacobabad",
        "precipitation": 0.4, "precip_3day_avg": 1.2, "precip_7day_avg": 0.9,
        "soil_moisture": 0.09, "soil_3day_avg": 0.11,
        "water_area_km2": 42.1, "water_area_change": -5.3, "water_area_pct_change": -0.11,
        "temperature": 43.7, "humidity": 24.0, "pressure": 99800.0,
        "evaporation": 0.0061, "wind_speed": 6.8,
        "month": 6, "day_of_year": 168, "is_monsoon": 0, "ds_idx": 1.0,
    },
    {
        "district": "Sindh_District",
        "precipitation": 387.0, "precip_3day_avg": 241.3, "precip_7day_avg": 138.6,
        "soil_moisture": 0.91, "soil_3day_avg": 0.87,
        "water_area_km2": 1420.7, "water_area_change": 640.2, "water_area_pct_change": 0.82,
        "temperature": 28.5, "humidity": 96.0, "pressure": 98200.0,
        "evaporation": -0.0031, "wind_speed": 5.2,
        "month": 9, "day_of_year": 258, "is_monsoon": 1, "ds_idx": 2.0,
    },
    {
        "district": "KP_District",
        "precipitation": 55.8, "precip_3day_avg": 38.4, "precip_7day_avg": 22.1,
        "soil_moisture": 0.42, "soil_3day_avg": 0.38,
        "water_area_km2": 320.9, "water_area_change": 45.6, "water_area_pct_change": 0.17,
        "temperature": 24.1, "humidity": 71.0, "pressure": 99100.0,
        "evaporation": -0.0009, "wind_speed": 3.3,
        "month": 7, "day_of_year": 196, "is_monsoon": 1, "ds_idx": 1.0,
    },
    {
        "district": "Balochistan_District",
        "precipitation": 18.3, "precip_3day_avg": 9.7, "precip_7day_avg": 5.4,
        "soil_moisture": 0.21, "soil_3day_avg": 0.19,
        "water_area_km2": 88.4, "water_area_change": 12.1, "water_area_pct_change": 0.16,
        "temperature": 33.8, "humidity": 41.0, "pressure": 99400.0,
        "evaporation": 0.0022, "wind_speed": 5.9,
        "month": 8, "day_of_year": 215, "is_monsoon": 1, "ds_idx": 4.0,
    },
]

FEATURE_COLS = [
    "precipitation", "precip_3day_avg", "precip_7day_avg",
    "soil_moisture", "soil_3day_avg",
    "water_area_km2", "water_area_change", "water_area_pct_change",
    "temperature", "humidity", "pressure", "evaporation", "wind_speed",
    "month", "day_of_year", "is_monsoon", "ds_idx",
]


def prob_to_risk(prob):
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def load_model(path):
    if not os.path.exists(path):
        print(f"\n  ERROR: Model file not found at '{path}'")
        print("  Update MODEL_PATH in the USER CONFIG section at the top of this file.")
        sys.exit(1)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"\n  ERROR: Could not load model from '{path}'")
        print(f"  Details: {e}")
        sys.exit(1)


def get_top_feature(model, feature_names):
    """Extract the most important feature from the model if available."""
    try:
        importances = model.feature_importances_
        idx = int(np.argmax(importances))
        return feature_names[idx]
    except AttributeError:
        pass
    try:
        # Linear models
        coef = np.abs(model.coef_[0])
        idx = int(np.argmax(coef))
        return feature_names[idx]
    except AttributeError:
        return "Not available"


def get_top_feature_from_artifacts(artifacts_path="artifacts/feature_importance.csv"):
    """Try to read a precomputed feature importance CSV from the artifacts folder.
    The CSV is expected to have columns: feature,importance and be ordered by importance.
    """
    try:
        if not os.path.exists(artifacts_path):
            return None
        df = pd.read_csv(artifacts_path)
        if df.empty:
            return None
        feat = str(df.iloc[0]["feature"])
        # strip common prefixes used in our pipeline exports
        for prefix in ("num__", "cat__", "feature__"):
            if feat.startswith(prefix):
                feat = feat[len(prefix) :]
                break
        # for categorical long names, keep until first comma if present
        if "," in feat:
            feat = feat.split(",")[0]
        return feat
    except Exception:
        return None


def validate_config():
    errors = []
    if TEAM_NAME == "YOUR_TEAM_NAME":
        errors.append("  - Set your TEAM_NAME in the USER CONFIG section.")
    if INSTITUTION == "YOUR_INSTITUTION":
        errors.append("  - Set your INSTITUTION in the USER CONFIG section.")
    if errors:
        print("\n  Please complete the USER CONFIG section before running:\n")
        for e in errors:
            print(e)
        sys.exit(1)


def main():
    print("\n" + "="*60)
    print("  FLOODSENSE — Prediction Leaderboard")
    print("  Neural Nova · BTech Fest 2026")
    print("="*60)

    validate_config()

    print(f"\n  Team       : {TEAM_NAME}")
    print(f"  Institution: {INSTITUTION}")
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Loading model...")

    model = load_model(MODEL_PATH)
    print("  Model loaded successfully.\n")

    df = pd.DataFrame(SCENARIOS)
    X = df[FEATURE_COLS].copy()

    # Ensure commonly derived/auxiliary features exist so older/newer models work
    # These mirror how the main code constructs them when unavailable.
    if "rain_soil_interaction" not in X.columns:
        X["rain_soil_interaction"] = X["precipitation"] * X["soil_moisture"]
    if "monsoon_cumulative_precip" not in X.columns:
        X["monsoon_cumulative_precip"] = np.where(X["is_monsoon"] == 1, X["precipitation"], 0.0)
    if "avg_elevation_m" not in X.columns:
        X["avg_elevation_m"] = 0.0
    if "water_area_acceleration" not in X.columns:
        X["water_area_acceleration"] = 0.0
    if "terrain_type" not in X.columns:
        X["terrain_type"] = df["district"].apply(lambda d: "Unknown")
    if "district" not in X.columns:
        X["district"] = df["district"]

    # Attempt to get the feature used for the model's training
    # (teams may have used a subset — handle gracefully)
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"\n  ERROR: Model prediction failed.\n  Details: {e}")
        print("\n  Common cause: your model was trained on different features.")
        print("  Check that your feature columns match exactly what you trained on.")
        sys.exit(1)

    top_feature = get_top_feature(model, FEATURE_COLS)
    if top_feature == "Not available":
        alt = get_top_feature_from_artifacts()
        if alt:
            top_feature = alt

    results = []
    print(f"  {'District':<25} {'Risk':<10} {'Confidence':>10}   Top Feature")
    print("  " + "-"*65)

    for i, scenario in enumerate(SCENARIOS):
        prob = float(probs[i])
        risk = prob_to_risk(prob)
        confidence = round(prob * 100, 1)

        print(f"  {scenario['district']:<25} {risk:<10} {confidence:>9.1f}%   {top_feature}")

        results.append({
            "team_name": TEAM_NAME,
            "institution": INSTITUTION,
            "district": scenario["district"],
            "risk_level": risk,
            "confidence_pct": confidence,
            "top_feature": top_feature,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    output_df = pd.DataFrame(results)
    output_file = "floodsense_submission.csv"
    output_df.to_csv(output_file, index=False)

    print("\n" + "="*60)
    print(f"  Submission saved: {output_file}")
    print("  Share this file via the Google Form before 12:10 PM.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
