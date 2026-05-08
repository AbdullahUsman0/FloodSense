from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import sys

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.config import FEATURES_USED, INSUFFICIENT_DATA_MESSAGE  # noqa: E402
from floodsense.inference import load_artifacts, predict_from_user_inputs  # noqa: E402


st.set_page_config(page_title="FloodSense Nexus", page_icon="🌊", layout="wide")
ARTIFACTS_DIR = ROOT / "artifacts"
BACKEND_URL = os.getenv("FLOODSENSE_BACKEND_URL", "http://127.0.0.1:8000")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
            :root {
                --bg-1: #040711;
                --bg-2: #0b1f2e;
                --card: rgba(8, 24, 38, 0.62);
                --line: rgba(123, 217, 255, 0.35);
                --glow: #1ec8ff;
                --text: #d9f2ff;
                --muted: #9ec9de;
            }
            .stApp {
                font-family: "IBM Plex Sans", sans-serif;
                color: var(--text);
                background:
                    radial-gradient(1200px 600px at 15% -10%, rgba(30, 200, 255, 0.25), transparent 60%),
                    radial-gradient(900px 500px at 90% 0%, rgba(80, 255, 200, 0.16), transparent 60%),
                    linear-gradient(160deg, var(--bg-1), var(--bg-2) 55%, #07131f);
            }
            .nexus-hero {
                border: 1px solid var(--line);
                border-radius: 18px;
                background: linear-gradient(140deg, rgba(10,31,47,.85), rgba(6,19,31,.76));
                padding: 20px 24px;
                box-shadow: 0 0 0 1px rgba(30,200,255,.15), 0 25px 45px rgba(0,0,0,.35);
                margin-bottom: 14px;
            }
            .nexus-title {
                font-family: "Space Grotesk", sans-serif;
                font-size: 34px;
                font-weight: 700;
                letter-spacing: .6px;
                margin: 0;
            }
            .nexus-subtitle {
                color: var(--muted);
                font-size: 15px;
                margin-top: 6px;
            }
            .glass-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 16px;
                background: var(--card);
                backdrop-filter: blur(8px);
            }
            .risk-badge {
                border-radius: 14px;
                padding: 14px 16px;
                color: #ffffff;
                font-weight: 700;
                letter-spacing: .4px;
                text-align: center;
                box-shadow: 0 0 24px rgba(14, 235, 255, 0.2);
                font-size: 28px;
                font-family: "Space Grotesk", sans-serif;
            }
            .mono-chip {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 6px 12px;
                font-size: 12px;
                display: inline-block;
                margin-right: 8px;
                color: #c8e8f9;
                background: rgba(9, 24, 36, 0.62);
            }
            div.stButton > button {
                border: 1px solid rgba(120, 225, 255, 0.6);
                color: #022036;
                background: linear-gradient(130deg, #7be8ff, #5dffc2);
                border-radius: 12px;
                font-weight: 700;
                transition: all .2s ease;
            }
            div.stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 8px 20px rgba(46, 218, 255, 0.25);
            }
            [data-testid="stSidebar"] {
                background: rgba(6, 19, 31, 0.82);
                border-right: 1px solid var(--line);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load_local_bundle():
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(ttl=60)
def _backend_get(path: str) -> dict:
    response = requests.get(f"{BACKEND_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def _backend_predict(payload: dict) -> dict:
    response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def _district_options() -> list[str]:
    try:
        return sorted(_backend_get("/districts").get("districts", []))
    except Exception:
        _, meta = _load_local_bundle()
        return sorted(meta.get("allowed_districts", []))


def _feature_list() -> list[str]:
    try:
        return _backend_get("/features").get("features_used", FEATURES_USED)
    except Exception:
        return FEATURES_USED


def _predict(payload: dict) -> dict:
    try:
        return _backend_predict(payload)
    except Exception:
        model, meta = _load_local_bundle()
        local = predict_from_user_inputs(
            model=model,
            meta=meta,
            rainfall_mm=payload["rainfall_mm"],
            selected_date=date.fromisoformat(payload["selected_date"]),
            district=payload["district"],
            soil_condition=payload["soil_condition"],
            visible_water=payload["visible_water"],
        )
        return local


def _render_header() -> None:
    st.markdown(
        """
        <div class="nexus-hero">
            <p class="nexus-title">FloodSense Nexus Console</p>
            <p class="nexus-subtitle">Bilingual district warning panel built for monsoon surge decisions</p>
            <span class="mono-chip">Low-Bandwidth Ready</span>
            <span class="mono-chip">English + Urdu</span>
            <span class="mono-chip">Extreme-Input Safe</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    _render_header()

    if not ARTIFACTS_DIR.exists():
        st.error("System not ready yet. Run training first.")
        return

    with st.sidebar:
        st.subheader("Features in Use")
        for item in _feature_list():
            st.write(f"- {item}")
        st.info(
            "If one rain sensor fails, the system fills that reading using the average rainfall from the two nearest districts."
        )
        st.caption(f"Backend URL: {BACKEND_URL}")

    col_left, col_right = st.columns([1.1, 1.0], gap="large")
    districts = _district_options()
    if not districts:
        st.error("No districts available in artifacts.")
        return

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Field Inputs")
        rainfall_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
        selected_date = st.date_input("Date", value=date.today())
        district = st.selectbox("District", options=districts)
        soil_condition = st.select_slider("Soil condition", options=["Dry", "Moist", "Saturated"], value="Moist")
        visible_water = st.radio("Visible surface water", options=["Yes", "No"], horizontal=True)
        go = st.button("Run Flood Check", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Decision Output")
        if not go:
            st.info("Enter inputs and run flood check.")
        else:
            payload = {
                "rainfall_mm": float(rainfall_mm),
                "selected_date": selected_date.isoformat(),
                "district": district,
                "soil_condition": soil_condition,
                "visible_water": visible_water,
            }
            result = _predict(payload)
            if not result.get("ok", False):
                st.warning(result.get("message", INSUFFICIENT_DATA_MESSAGE))
            else:
                st.markdown(
                    f'<div class="risk-badge" style="background:{result["risk_color"]};">'
                    f'{result["risk_level_en"]} / {result["risk_level_ur"]}</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Confidence / اعتماد", f'{result["confidence_pct"]:.1f}%')
                st.metric("Estimated people at risk / متوقع خطرے میں آبادی", f'{result["population_risk_estimate"]:,}')
                st.write("**Recommended action / تجویز کردہ کارروائی**")
                st.write(f'- {result["recommended_action_en"]}')
                st.write(f'- {result["recommended_action_ur"]}')
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
