# FloodSense Delivery Notes

## Features Used
- Rainfall: `precipitation`, `precip_3day_avg`, `precip_7day_avg`
- Soil: `soil_moisture`, `soil_3day_avg`
- Water extent: `water_area_km2`, `water_area_change`, `water_area_pct_change`
- Atmospheric: `temperature`, `humidity`, `pressure`, `evaporation`
- Temporal: `month`, `day_of_year`, `is_monsoon`
- Flood history: `ds_idx`
- Terrain merge: `avg_elevation_m` from `district_elevation_reference.csv`

## Scenario Card 1 (Monsoon Surge)
- Enabled via training flag: `--apply-scenario-card1`
- Adds two new rows into training data for impacted districts with a 300% rainfall increase.
- Writes augmented dataset copy to:
  - `artifacts/scenario_card1_augmented_training_data.csv`
- Retrains model on augmented dataset.
- Validates all districts still return valid risk labels under extreme rainfall.

## Scenario Card 2 (Sensor Went Rogue)
- Enabled via training flag: `--apply-scenario-card2 --faulty-district Sindh_District`
- Keeps the district in the dataset.
- Replaces the faulty district rainfall value for the current cycle with:
  - average rainfall from two nearest districts (based on station-coordinate proximity in the dataset)
- Writes imputed dataset copy to:
  - `artifacts/scenario_card2_imputed_training_data.csv`
- Frontend includes plain-language note for officials describing this fallback.

## Local Run (No Docker)
1. Install dependencies:
   - `.\.python\cpython-3.11.15-windows-x86_64-none\python.exe -m pip install --break-system-packages -r requirements.txt`
2. Train with Scenario Card 1 + Scenario Card 2:
   - `.\.python\cpython-3.11.15-windows-x86_64-none\python.exe scripts/train_and_evaluate.py --data-dir . --artifacts-dir artifacts --apply-scenario-card2 --faulty-district Sindh_District --apply-scenario-card1 --impacted-districts Sindh_District KP_District --scenario-spike-increase-pct 300`
3. Start backend:
   - `.\.python\cpython-3.11.15-windows-x86_64-none\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
4. Start frontend (new terminal):
   - `$env:FLOODSENSE_BACKEND_URL='http://127.0.0.1:8000'`
   - `.\.python\cpython-3.11.15-windows-x86_64-none\python.exe -m streamlit run app/streamlit_app.py`

## Docker Hosting
1. Build and run both services:
   - `docker compose up --build`
2. Open frontend:
   - `http://localhost:8501`
3. Backend health endpoint:
   - `http://localhost:8000/health`

## Main Outputs
- `artifacts/model.joblib`
- `artifacts/model_meta.json`
- `artifacts/feature_importance.csv`
- `artifacts/training_report.md`
- `artifacts/district_extreme_risk_validation.json`
- `artifacts/scenario_card1_augmented_training_data.csv` (when scenario flag used)
- `artifacts/scenario_card2_imputed_training_data.csv` (when scenario flag used)
