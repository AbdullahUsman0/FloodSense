from __future__ import annotations

from dataclasses import dataclass


LOW_THRESHOLD = 0.25
MEDIUM_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.75
INSUFFICIENT_DATA_MESSAGE = "Insufficient data — manual assessment recommended"


RISK_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#EF6C00",
    "Critical": "#C62828",
}


RISK_LABELS_UR = {
    "Low": "کم",
    "Medium": "درمیانہ",
    "High": "زیادہ",
    "Critical": "انتہائی",
}


RISK_ACTIONS = {
    "Low": {
        "en": "Continue routine monitoring and keep village contact lists updated.",
        "ur": "معمول کی نگرانی جاری رکھیں اور دیہی رابطہ فہرستیں تازہ رکھیں۔",
    },
    "Medium": {
        "en": "Alert union councils and prepare evacuation transport standby.",
        "ur": "یونین کونسلز کو الرٹ کریں اور انخلا کے لیے ٹرانسپورٹ تیار رکھیں۔",
    },
    "High": {
        "en": "Issue district-level warning and pre-position rescue teams near high-risk zones.",
        "ur": "ضلعی وارننگ جاری کریں اور ریسکیو ٹیموں کو بلند خطرے والے علاقوں کے قریب تعینات کریں۔",
    },
    "Critical": {
        "en": "Start immediate evacuation in vulnerable settlements and activate emergency shelters.",
        "ur": "کمزور آبادیوں میں فوری انخلا شروع کریں اور ہنگامی پناہ گاہیں فعال کریں۔",
    },
}


FEATURES_USED = [
    "Rainfall: precipitation, precip_3day_avg, precip_7day_avg",
    "Soil: soil_moisture, soil_3day_avg",
    "Water extent: water_area_km2, water_area_change, water_area_pct_change",
    "Atmospheric: temperature, humidity, pressure, evaporation",
    "Temporal: month, day_of_year, is_monsoon",
    "Engineered: rain_soil_interaction, monsoon_cumulative_precip, water_area_acceleration",
    "Terrain: avg_elevation_m (from district_elevation_reference.csv)",
]


DISTRICT_TO_REGION = {
    "Sindh_District": "Sindh",
    "Jacobabad": "Sindh",
    "KP_District": "KP",
    "Nowshera": "KP",
    "Balochistan_District": "Balochistan",
}

DISTRICT_RISK_PROFILE = {
    "Sindh_District": {"settlement_type": "rural_floodplain", "exposure_factor": 0.85, "max_exposed_share": 0.08},
    "Jacobabad": {"settlement_type": "urban_floodplain", "exposure_factor": 0.95, "max_exposed_share": 0.09},
    "KP_District": {"settlement_type": "mixed_riverine", "exposure_factor": 0.80, "max_exposed_share": 0.07},
    "Nowshera": {"settlement_type": "urban_riverine", "exposure_factor": 0.90, "max_exposed_share": 0.08},
    "Balochistan_District": {"settlement_type": "sparse_plateau", "exposure_factor": 0.55, "max_exposed_share": 0.05},
}


REGION_DISTRICT_DIVISOR = {
    "Sindh": 2,
    "KP": 2,
    "Balochistan": 1,
}


SOIL_CONDITION_TO_VALUE = {
    "Dry": 0.20,
    "Moist": 0.45,
    "Saturated": 0.70,
}


VISIBLE_WATER_OPTIONS = {"Yes", "No"}


MODEL_NUMERIC_FEATURES = [
    "precipitation",
    "precip_3day_avg",
    "precip_7day_avg",
    "soil_moisture",
    "soil_3day_avg",
    "water_area_km2",
    "water_area_change",
    "water_area_pct_change",
    "temperature",
    "humidity",
    "pressure",
    "evaporation",
    "month",
    "day_of_year",
    "is_monsoon",
    "rain_soil_interaction",
    "monsoon_cumulative_precip",
    "water_area_acceleration",
    "avg_elevation_m",
]


MODEL_CATEGORICAL_FEATURES = [
    "district",
    "terrain_type",
]


DROP_COLUMNS = [
    "flood_event",
    "date",
    "elevation",
    "latitude",
    "longitude",
]


OUTLIER_BOUNDS = {
    "soil_moisture": (0.0, 1.0),
    "humidity": (0.0, 100.0),
    "temperature": (-20.0, 60.0),
    "precipitation": (0.0, 500.0),
    "water_area_km2": (0.0, None),
    "wind_speed": (0.0, None),
}


@dataclass(frozen=True)
class RiskResult:
    risk_level_en: str
    risk_level_ur: str
    confidence_pct: float
    recommended_action_en: str
    recommended_action_ur: str
    population_risk_estimate: int
