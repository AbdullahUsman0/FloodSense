from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScenarioInjectedRow:
    district: str
    original_date: str
    scenario_date: str
    original_precipitation: float
    scenario_precipitation: float
    spike_multiplier_applied: float


@dataclass
class ScenarioSensorRogueImputation:
    faulty_district: str
    faulty_date: str
    original_precipitation: float | None
    imputed_precipitation: float
    donor_districts: list[str]
    donor_values: list[float]


def apply_scenario_card1_monsoon_surge(
    train_df: pd.DataFrame,
    impacted_districts: tuple[str, str] = ("Sindh_District", "KP_District"),
    rainfall_spike_increase_pct: float = 300.0,
) -> tuple[pd.DataFrame, list[ScenarioInjectedRow]]:
    """
    Scenario Card 1:
    - Add new rows for two impacted districts with a 300% rainfall spike.
    - Interprets 300% spike as +300% increase over baseline (4x original).
    """
    working = train_df.copy()
    working["_parsed_date"] = pd.to_datetime(working["date"], format="%m/%d/%Y", errors="coerce")

    increase_factor = 1.0 + (rainfall_spike_increase_pct / 100.0)
    added_rows: list[pd.Series] = []
    injected: list[ScenarioInjectedRow] = []

    latest_global_date = working["_parsed_date"].max()
    if pd.isna(latest_global_date):
        latest_global_date = pd.Timestamp("2024-12-31")
    offset_days = 1

    for district in impacted_districts:
        district_rows = working[(working["district"] == district) & working["_parsed_date"].notna()].sort_values("_parsed_date")
        if district_rows.empty:
            continue

        precip_candidates = district_rows.copy()
        precip_candidates["_precip"] = pd.to_numeric(precip_candidates["precipitation"], errors="coerce")
        precip_candidates = precip_candidates[precip_candidates["_precip"].notna()]
        if not precip_candidates.empty:
            source = precip_candidates.sort_values("_precip", ascending=False).iloc[0].copy()
        else:
            source = district_rows.iloc[-1].copy()
        source_date = source["_parsed_date"]
        scenario_date = latest_global_date + pd.Timedelta(days=offset_days)
        offset_days += 1

        original_precip = pd.to_numeric(source.get("precipitation"), errors="coerce")
        if pd.isna(original_precip) or original_precip <= 0:
            fallback = pd.to_numeric(district_rows["precip_7day_avg"], errors="coerce").dropna()
            original_precip = float(fallback.iloc[-1]) if not fallback.empty else 50.0
        else:
            original_precip = float(original_precip)

        scenario_precip = min(500.0, original_precip * increase_factor)
        source["precipitation"] = scenario_precip
        source["precip_3day_avg"] = min(500.0, (0.35 * pd.to_numeric(source.get("precip_3day_avg"), errors="coerce")) + (0.65 * scenario_precip))
        source["precip_7day_avg"] = min(500.0, (0.55 * pd.to_numeric(source.get("precip_7day_avg"), errors="coerce")) + (0.45 * scenario_precip))

        source["soil_moisture"] = min(1.0, float(pd.to_numeric(source.get("soil_moisture"), errors="coerce")) + 0.12)
        source["soil_3day_avg"] = min(1.0, float(pd.to_numeric(source.get("soil_3day_avg"), errors="coerce")) + 0.08)
        source["water_area_km2"] = max(0.0, float(pd.to_numeric(source.get("water_area_km2"), errors="coerce")) * 1.45)
        source["water_area_change"] = max(1.0, float(pd.to_numeric(source.get("water_area_change"), errors="coerce")) + 3.5)
        prev_water = max(source["water_area_km2"] - source["water_area_change"], 0.001)
        source["water_area_pct_change"] = source["water_area_change"] / prev_water
        source["flood_event"] = 1

        source["_parsed_date"] = scenario_date
        source["date"] = f"{scenario_date.month}/{scenario_date.day}/{scenario_date.year}"
        source = source.drop(labels=["_precip"], errors="ignore")
        added_rows.append(source)
        injected.append(
            ScenarioInjectedRow(
                district=district,
                original_date=f"{source_date.month}/{source_date.day}/{source_date.year}",
                scenario_date=source["date"],
                original_precipitation=round(original_precip, 3),
                scenario_precipitation=round(float(scenario_precip), 3),
                spike_multiplier_applied=round(increase_factor, 3),
            )
        )

    if added_rows:
        augmented = pd.concat([working, pd.DataFrame(added_rows)], ignore_index=True)
    else:
        augmented = working
    augmented = augmented.drop(columns=["_parsed_date"], errors="ignore")
    return augmented, injected


def apply_scenario_card2_sensor_rogue(
    train_df: pd.DataFrame,
    faulty_district: str = "Sindh_District",
) -> tuple[pd.DataFrame, ScenarioSensorRogueImputation | None]:
    """
    Scenario Card 2:
    - One district's current-cycle rainfall sensor is faulty.
    - Keep district in data and impute rainfall from average of two nearest districts.
    - Proximity is based on district station coordinates in dataset.
    """
    working = train_df.copy()
    working["_parsed_date"] = pd.to_datetime(working["date"], format="%m/%d/%Y", errors="coerce")
    if working["_parsed_date"].isna().all():
        return train_df.copy(), None

    district_rows = working[(working["district"] == faulty_district) & working["_parsed_date"].notna()].sort_values("_parsed_date")
    if district_rows.empty:
        return train_df.copy(), None

    target_idx = district_rows.index[-1]
    target_date = working.loc[target_idx, "_parsed_date"]

    # Compute district coordinate centroids.
    coord_df = working.copy()
    coord_df["latitude"] = pd.to_numeric(coord_df["latitude"], errors="coerce")
    coord_df["longitude"] = pd.to_numeric(coord_df["longitude"], errors="coerce")
    centroids = (
        coord_df.dropna(subset=["latitude", "longitude"])
        .groupby("district", as_index=False)[["latitude", "longitude"]]
        .mean()
    )
    target_row = centroids[centroids["district"] == faulty_district]
    if target_row.empty:
        return train_df.copy(), None
    tlat = float(target_row.iloc[0]["latitude"])
    tlon = float(target_row.iloc[0]["longitude"])

    candidates = centroids[centroids["district"] != faulty_district].copy()
    candidates["dist"] = np.sqrt((candidates["latitude"] - tlat) ** 2 + (candidates["longitude"] - tlon) ** 2)
    donor_districts = candidates.sort_values("dist").head(2)["district"].tolist()
    if len(donor_districts) < 2:
        return train_df.copy(), None

    donor_values: list[float] = []
    for donor in donor_districts:
        donor_rows = working[(working["district"] == donor) & working["_parsed_date"].notna()].copy()
        if donor_rows.empty:
            continue
        donor_rows["precipitation"] = pd.to_numeric(donor_rows["precipitation"], errors="coerce")
        same_date = donor_rows[donor_rows["_parsed_date"] == target_date]
        if not same_date.empty and same_date["precipitation"].notna().any():
            donor_values.append(float(same_date["precipitation"].dropna().iloc[-1]))
            continue
        donor_rows["date_gap"] = (donor_rows["_parsed_date"] - target_date).abs()
        donor_rows = donor_rows.sort_values("date_gap")
        nearest_valid = donor_rows["precipitation"].dropna()
        if not nearest_valid.empty:
            donor_values.append(float(nearest_valid.iloc[0]))

    if len(donor_values) < 2:
        global_precip = pd.to_numeric(working["precipitation"], errors="coerce").dropna()
        fallback = float(global_precip.median()) if not global_precip.empty else 0.0
        while len(donor_values) < 2:
            donor_values.append(fallback)

    imputed_precip = float(np.mean(donor_values[:2]))
    original_precip = pd.to_numeric(working.loc[target_idx, "precipitation"], errors="coerce")
    working.loc[target_idx, "precipitation"] = imputed_precip

    # Keep rolling fields coherent with imputed value.
    for col, alpha in [("precip_3day_avg", 0.6), ("precip_7day_avg", 0.35)]:
        existing = pd.to_numeric(working.loc[target_idx, col], errors="coerce")
        if pd.isna(existing):
            working.loc[target_idx, col] = imputed_precip
        else:
            working.loc[target_idx, col] = (alpha * imputed_precip) + ((1.0 - alpha) * float(existing))

    details = ScenarioSensorRogueImputation(
        faulty_district=faulty_district,
        faulty_date=f"{target_date.month}/{target_date.day}/{target_date.year}",
        original_precipitation=None if pd.isna(original_precip) else float(original_precip),
        imputed_precipitation=round(imputed_precip, 3),
        donor_districts=donor_districts,
        donor_values=[round(v, 3) for v in donor_values[:2]],
    )
    working = working.drop(columns=["_parsed_date"], errors="ignore")
    return working, details
