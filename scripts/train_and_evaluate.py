from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.config import MODEL_NUMERIC_FEATURES, RISK_ACTIONS, RISK_LABELS_UR, RiskResult
from floodsense.data_pipeline import (
    assert_no_non_finite_inputs,
    attach_elevation_features,
    build_district_month_baseline,
    clean_training_data,
    get_feature_target_frame,
    load_raw_data,
    time_based_split,
)
from floodsense.inference import (
    apply_operational_risk_guardrail,
    estimate_population_at_risk,
    format_result_payload,
    predict_risk,
)
from floodsense.modeling import choose_best_model, build_candidate_pipelines, evaluate_binary_classifier, extract_feature_importance
from floodsense.scenario import apply_scenario_card1_monsoon_surge
from floodsense.scenario import apply_scenario_card2_sensor_rogue


def _metrics_to_dict(result) -> dict:
    return {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "confusion_matrix": result.confusion_matrix,
    }


def _write_markdown_report(
    artifacts_dir: Path,
    selected_model_name: str,
    scorecard: dict,
    quality_stats,
    top_features: pd.DataFrame,
    scenario_details: dict | None,
    scenario2_details: dict | None,
    district_audit: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# FloodSense Training Report")
    lines.append("")
    lines.append("## Data Quality Summary")
    lines.append(f"- Rows before cleanup: **{quality_stats.rows_before}**")
    lines.append(f"- Rows after cleanup: **{quality_stats.rows_after}**")
    lines.append(f"- Exact duplicates removed: **{quality_stats.duplicates_removed}**")
    lines.append(f"- Phantom rows removed: **{quality_stats.phantom_rows_removed}**")
    lines.append(f"- Invalid date rows removed: **{quality_stats.bad_date_rows_removed}**")
    lines.append(f"- Missing precipitation values handled: **{quality_stats.precipitation_missing_count}**")
    lines.append(f"- Non-finite water % values handled: **{quality_stats.water_pct_non_finite_count}**")
    lines.append("")
    lines.append("## Model Selection")
    lines.append(f"- Selected model: **{selected_model_name}**")
    lines.append("")
    lines.append("## Scenario Card 1")
    if scenario_details:
        lines.append(f"- Applied: **{scenario_details.get('applied', False)}**")
        lines.append(f"- Spike increase configured: **{scenario_details.get('rainfall_spike_increase_pct', 300)}%**")
        lines.append(f"- Affected districts: **{', '.join(scenario_details.get('impacted_districts', []))}**")
        lines.append(f"- Injected rows: **{len(scenario_details.get('injected_rows', []))}**")
        for row in scenario_details.get("injected_rows", []):
            lines.append(
                f"- {row['district']}: {row['original_precipitation']} -> {row['scenario_precipitation']} mm "
                f"on {row['scenario_date']} (x{row['spike_multiplier_applied']})"
            )
    else:
        lines.append("- Not applied.")
    lines.append("")

    lines.append("## Scenario Card 2")
    if scenario2_details:
        lines.append(f"- Applied: **{scenario2_details.get('applied', False)}**")
        lines.append(f"- Faulty district: **{scenario2_details.get('faulty_district')}**")
        lines.append(f"- Faulty cycle date: **{scenario2_details.get('faulty_date')}**")
        lines.append(f"- Nearest donor districts: **{', '.join(scenario2_details.get('donor_districts', []))}**")
        lines.append(
            f"- Rainfall replaced: **{scenario2_details.get('original_precipitation')} -> "
            f"{scenario2_details.get('imputed_precipitation')} mm**"
        )
    else:
        lines.append("- Not applied.")
    lines.append("")

    lines.append("## District Extreme-Value Validation")
    lines.append("- Required check: every district returns a non-null risk class under extreme rainfall input.")
    for row in district_audit:
        lines.append(
            f"- {row['district']}: {row['risk_level_en']} / {row['risk_level_ur']} "
            f"(confidence={row['confidence_pct']}%, rainfall={row['rainfall_mm']} mm)"
        )
    lines.append("")

    lines.append("## Evaluation Metrics")
    for name, splits in scorecard.items():
        lines.append(f"### {name}")
        for split_name, metrics in splits.items():
            lines.append(f"- {split_name}: accuracy={metrics.accuracy:.4f}, precision={metrics.precision:.4f}, recall={metrics.recall:.4f}, f1={metrics.f1:.4f}")
            lines.append(f"  confusion_matrix={metrics.confusion_matrix}")
        lines.append("")

    lines.append("## Top Feature Importance")
    if top_features.empty:
        lines.append("- Not available for this model.")
    else:
        for _, row in top_features.head(15).iterrows():
            lines.append(f"- {row['feature']}: {row['importance']:.5f}")

    (artifacts_dir / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FloodSense model and build artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--apply-scenario-card1", action="store_true", help="Inject 300% rainfall surge rows for two districts.")
    parser.add_argument("--impacted-districts", nargs=2, default=["Sindh_District", "KP_District"])
    parser.add_argument("--scenario-spike-increase-pct", type=float, default=300.0)
    parser.add_argument("--apply-scenario-card2", action="store_true", help="Impute one faulty district rainfall from two nearest districts.")
    parser.add_argument("--faulty-district", type=str, default="Sindh_District")
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df, elevation_df, ndma_df = load_raw_data(args.data_dir)

    scenario_details: dict | None = None
    scenario2_details: dict | None = None
    if args.apply_scenario_card2:
        imputed_df, detail = apply_scenario_card2_sensor_rogue(
            train_df=train_df,
            faulty_district=args.faulty_district,
        )
        if detail is not None:
            scenario2_details = {"applied": True, **vars(detail)}
            train_df = imputed_df
            imputed_df.to_csv(args.artifacts_dir / "scenario_card2_imputed_training_data.csv", index=False)

    if args.apply_scenario_card1:
        augmented_df, injected = apply_scenario_card1_monsoon_surge(
            train_df=train_df,
            impacted_districts=(args.impacted_districts[0], args.impacted_districts[1]),
            rainfall_spike_increase_pct=args.scenario_spike_increase_pct,
        )
        scenario_details = {
            "applied": True,
            "rainfall_spike_increase_pct": args.scenario_spike_increase_pct,
            "impacted_districts": args.impacted_districts,
            "injected_rows": [vars(item) for item in injected],
        }
        train_df = augmented_df
        augmented_df.to_csv(args.artifacts_dir / "scenario_card1_augmented_training_data.csv", index=False)
    cleaned_df, quality_stats = clean_training_data(train_df)
    model_df = attach_elevation_features(cleaned_df, elevation_df)

    X, y = get_feature_target_frame(model_df)
    assert_no_non_finite_inputs(X, MODEL_NUMERIC_FEATURES)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train_t, X_test_t, y_train_t, y_test_t = time_based_split(X, y, cleaned_df["date"], test_fraction=0.2)

    candidates = build_candidate_pipelines(random_state=42)
    scorecard = {}

    for model_name, pipeline in candidates.items():
        pipeline_s = copy.deepcopy(pipeline)
        pipeline_s.fit(X_train_s, y_train_s)
        eval_s = evaluate_binary_classifier(pipeline_s, X_test_s, y_test_s)

        pipeline_t = copy.deepcopy(pipeline)
        pipeline_t.fit(X_train_t, y_train_t)
        eval_t = evaluate_binary_classifier(pipeline_t, X_test_t, y_test_t)
        scorecard[model_name] = {
            "stratified": eval_s,
            "time_based": eval_t,
        }

    best_name = choose_best_model(scorecard)
    final_pipeline = candidates[best_name]
    final_pipeline.fit(X, y)
    proba = final_pipeline.predict_proba(X)[:, 1]

    feature_importance = extract_feature_importance(final_pipeline)
    feature_importance.to_csv(args.artifacts_dir / "feature_importance.csv", index=False)

    ndma_population_map = {
        str(row["region"]): int(row["affected_population"])
        for _, row in ndma_df.iterrows()
        if not pd.isna(row["affected_population"])
    }
    baseline_payload = build_district_month_baseline(model_df)
    for _, row in elevation_df.iterrows():
        district_name = str(row["district"])
        baseline_payload["terrain_lookup"].setdefault(
            district_name,
            {
                "terrain_type": str(row["terrain_type"]),
                "avg_elevation_m": float(row["avg_elevation_m"]),
            },
        )
    allowed_districts = sorted(set(model_df["district"].dropna().tolist()) | set(elevation_df["district"].dropna().tolist()))

    metadata = {
        "selected_model": best_name,
        "quality_stats": vars(quality_stats),
        "metrics": {
            model_name: {
                split_name: _metrics_to_dict(metrics)
                for split_name, metrics in split_results.items()
            }
            for model_name, split_results in scorecard.items()
        },
        "flood_probability_summary": {
            "min": float(np.min(proba)),
            "max": float(np.max(proba)),
            "mean": float(np.mean(proba)),
        },
        "ndma_population_by_region": ndma_population_map,
        "baseline_payload": baseline_payload,
        "allowed_districts": allowed_districts,
        "scenario_card1": scenario_details or {"applied": False},
        "scenario_card2": scenario2_details or {"applied": False},
    }

    # Mandatory robustness check: extreme value classification for each district.
    district_audit: list[dict] = []
    default_date = cleaned_df["date"].max()
    if pd.isna(default_date):
        default_date = pd.Timestamp("2024-12-31")
    simulation_date = (default_date + timedelta(days=1)).date()
    for district in allowed_districts:
        key = f"{district}::{simulation_date.month}"
        base = baseline_payload["district_month"].get(key, baseline_payload["default"])
        terrain = baseline_payload["terrain_lookup"].get(district, {"terrain_type": "Unknown", "avg_elevation_m": base.get("avg_elevation_m", 0.0)})
        row_dict = dict(base)
        row_dict.update(
            {
                "district": district,
                "terrain_type": terrain["terrain_type"],
                "avg_elevation_m": float(terrain["avg_elevation_m"]),
                "month": simulation_date.month,
                "day_of_year": simulation_date.timetuple().tm_yday,
                "year": simulation_date.year,
                "is_monsoon": int(simulation_date.month in [7, 8, 9]),
                "precipitation": 500.0,
                "precip_3day_avg": min(500.0, float(base.get("precip_3day_avg", 120.0)) * 1.8),
                "precip_7day_avg": min(500.0, float(base.get("precip_7day_avg", 90.0)) * 1.6),
                "soil_moisture": min(1.0, float(base.get("soil_moisture", 0.5)) + 0.2),
                "soil_3day_avg": min(1.0, float(base.get("soil_3day_avg", 0.45)) + 0.15),
                "water_area_km2": max(0.0, float(base.get("water_area_km2", 5.0)) * 1.7),
                "water_area_change": max(1.0, float(base.get("water_area_change", 1.0)) + 3.0),
            }
        )
        prev_water = max(row_dict["water_area_km2"] - row_dict["water_area_change"], 0.001)
        row_dict["water_area_pct_change"] = row_dict["water_area_change"] / prev_water
        extreme_row = pd.DataFrame([row_dict])
        result = predict_risk(final_pipeline, extreme_row, district, ndma_population_map)
        forced_level = apply_operational_risk_guardrail(
            base_risk_level_en=result.risk_level_en,
            rainfall_mm=500.0,
            soil_condition="Saturated",
            visible_water="Yes",
        )
        if forced_level != result.risk_level_en:
            result = RiskResult(
                risk_level_en=forced_level,
                risk_level_ur=RISK_LABELS_UR[forced_level],
                confidence_pct=result.confidence_pct,
                recommended_action_en=RISK_ACTIONS[forced_level]["en"],
                recommended_action_ur=RISK_ACTIONS[forced_level]["ur"],
                population_risk_estimate=estimate_population_at_risk(
                    district=district,
                    ndma_population_by_region=ndma_population_map,
                    risk_level=forced_level,
                    confidence_pct=result.confidence_pct,
                ),
            )
        payload = format_result_payload(result)
        district_audit.append(
            {
                "district": district,
                "rainfall_mm": row_dict["precipitation"],
                "risk_level_en": payload["risk_level_en"],
                "risk_level_ur": payload["risk_level_ur"],
                "confidence_pct": payload["confidence_pct"],
            }
        )

    for item in district_audit:
        if item["risk_level_en"] not in {"Low", "Medium", "High", "Critical"}:
            raise ValueError(f"Invalid classification for district {item['district']}: {item['risk_level_en']}")

    joblib.dump(final_pipeline, args.artifacts_dir / "model.joblib")
    with (args.artifacts_dir / "model_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (args.artifacts_dir / "district_extreme_risk_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(district_audit, handle, indent=2, ensure_ascii=False)

    model_df.to_csv(args.artifacts_dir / "cleaned_training_data.csv", index=False)
    _write_markdown_report(
        artifacts_dir=args.artifacts_dir,
        selected_model_name=best_name,
        scorecard=scorecard,
        quality_stats=quality_stats,
        top_features=feature_importance,
        scenario_details=scenario_details,
        scenario2_details=scenario2_details,
        district_audit=district_audit,
    )

    selected_split_metrics = scorecard[best_name]
    strat_acc = selected_split_metrics["stratified"].accuracy
    time_acc = selected_split_metrics["time_based"].accuracy
    print(f"Selected model: {best_name}")
    print(f"Stratified accuracy: {strat_acc:.4f}")
    print(f"Time-based accuracy: {time_acc:.4f}")
    if strat_acc < 0.70 or time_acc < 0.70:
        print("WARNING: One split is below 0.70 accuracy. Review feature engineering and thresholds.")


if __name__ == "__main__":
    main()
