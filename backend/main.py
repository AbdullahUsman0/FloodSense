from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.config import FEATURES_USED, INSUFFICIENT_DATA_MESSAGE  # noqa: E402
from floodsense.inference import load_artifacts, predict_from_user_inputs  # noqa: E402


ARTIFACTS_DIR = ROOT / "artifacts"
app = FastAPI(title="FloodSense Backend", version="1.0.0")


class PredictRequest(BaseModel):
    rainfall_mm: float = Field(..., ge=0.0, le=500.0)
    selected_date: date
    district: str
    soil_condition: str
    visible_water: str


class PredictResponse(BaseModel):
    ok: bool
    message: str | None = None
    risk_level_en: str | None = None
    risk_level_ur: str | None = None
    confidence_pct: float | None = None
    recommended_action_en: str | None = None
    recommended_action_ur: str | None = None
    population_risk_estimate: int | None = None
    risk_color: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def feature_list() -> dict[str, list[str]]:
    return {"features_used": FEATURES_USED}


@app.get("/districts")
def districts() -> dict[str, list[str]]:
    _, meta = load_artifacts(ARTIFACTS_DIR)
    return {"districts": sorted(meta.get("allowed_districts", []))}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model, meta = load_artifacts(ARTIFACTS_DIR)
    result = predict_from_user_inputs(
        model=model,
        meta=meta,
        rainfall_mm=payload.rainfall_mm,
        selected_date=payload.selected_date,
        district=payload.district,
        soil_condition=payload.soil_condition,
        visible_water=payload.visible_water,
    )
    if not result.get("ok", False):
        return PredictResponse(ok=False, message=result.get("message", INSUFFICIENT_DATA_MESSAGE))
    return PredictResponse(**result)

