from __future__ import annotations

import json
import os
from datetime import date
from html import escape
from pathlib import Path
import sys

import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from floodsense.config import DISTRICT_RISK_PROFILE, FEATURES_USED, INSUFFICIENT_DATA_MESSAGE  # noqa: E402
from floodsense.inference import load_artifacts, predict_from_user_inputs  # noqa: E402
from floodsense.provincial_alerts import merge_provincial_alerts_by_same_advisory  # noqa: E402


st.set_page_config(page_title="FloodSense - Flood Risk", layout="centered")
ARTIFACTS_DIR = ROOT / "artifacts"
BACKEND_URL = os.getenv("FLOODSENSE_BACKEND_URL", "http://127.0.0.1:8000")


RISK_STYLES = {
    "Low": {"bg": "#D1FAE5", "text": "#065F46", "border": "#34D399"},
    "Medium": {"bg": "#FEF9C3", "text": "#78350F", "border": "#FBBF24"},
    "High": {"bg": "#FED7AA", "text": "#7C2D12", "border": "#F97316"},
    "Critical": {"bg": "#FEE2E2", "text": "#7F1D1D", "border": "#EF4444"},
}

URDU_RISK_LABELS = {
    "Low": "کم",
    "Medium": "درمیانہ",
    "High": "زیادہ",
    "Critical": "انتہائی",
}

URDU_ACTIONS = {
    "Low": "معمول کی نگرانی جاری رکھیں اور گاؤں کی رابطہ فہرستیں تازہ رکھیں۔",
    "Medium": "یونین کونسلز کو خبردار کریں اور انخلا کے لیے ٹرانسپورٹ تیار رکھیں۔",
    "High": "ضلعی سطح کی وارننگ جاری کریں اور امدادی ٹیموں کو خطرے والے علاقوں کے قریب رکھیں۔",
    "Critical": "کمزور آبادیوں میں فوری انخلا شروع کریں اور ہنگامی پناہ گاہیں فعال کریں۔",
}

URDU_TTS_MESSAGES = {
    "Low": "خطرہ کم ہے۔ نگرانی جاری رکھیں۔",
    "Medium": "احتیاط کریں۔ مقامی ٹیم کو تیار رکھیں۔",
    "High": "خطرہ زیادہ ہے۔ لوگوں کو خبردار کریں۔",
    "Critical": "انتہائی خطرہ ہے۔ محفوظ جگہ پر فوری انخلا کریں۔",
}

ACTION_LEVELS = {
    "Low": {"icon": "OK", "label": "Monitor", "urdu": "نگرانی کریں", "detail": "Continue routine checks."},
    "Medium": {"icon": "!", "label": "Prepare", "urdu": "تیاری کریں", "detail": "Alert local teams and prepare transport."},
    "High": {"icon": "!!", "label": "Warn", "urdu": "خبردار کریں", "detail": "Issue warning and position rescue teams."},
    "Critical": {"icon": "!!!", "label": "Evacuate", "urdu": "انخلا کریں", "detail": "Begin immediate evacuation in vulnerable areas."},
}

_CANONICAL_RISK_ORDER = frozenset({"Low", "Medium", "High", "Critical"})
_RISK_ALIASES = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}


def _normalize_risk_level(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s in _CANONICAL_RISK_ORDER:
        return s
    return _RISK_ALIASES.get(s.lower())


def _district_display_name(district_key: str) -> str:
    return district_key.strip().replace("_", " ")

EMERGENCY_CONTACTS = [
    {"label": "Rescue / Emergency", "value": "1122", "detail": "Emergency response and rescue"},
    {"label": "Police", "value": "15", "detail": "Law and order support"},
    {"label": "NDMA", "value": "1135", "detail": "National disaster management helpline"},
]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @font-face {
                font-family: 'Noto Nastaliq Urdu';
                src: url('/app/static/fonts/NotoNastaliqUrdu-Regular.woff2') format('woff2');
                font-weight: 400 700;
                font-style: normal;
                font-display: swap;
            }

            html, body, [class*="css"] {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            .stApp {
                background: #F5F7FA;
                color: #111827;
            }
            .block-container {
                max-width: 860px;
                padding-top: 24px;
                padding-bottom: 48px;
            }
            .main-card {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.07);
                padding: 22px;
                margin-bottom: 18px;
            }
            .header-title {
                margin: 0;
                color: #111827;
                font-size: 28px;
                line-height: 1.25;
                font-weight: 800;
                letter-spacing: 0;
            }
            .header-subtitle {
                margin: 8px 0 0 0;
                color: #4B5563;
                font-size: 17px;
                line-height: 1.5;
                font-weight: 500;
            }
            .section-title {
                margin: 0 0 18px 0;
                color: #111827;
                font-size: 22px;
                line-height: 1.3;
                font-weight: 700;
                letter-spacing: 0;
            }
            .field-label-en {
                color: #111827;
                font-size: 17px;
                line-height: 1.45;
                font-weight: 600;
                margin-bottom: 2px;
            }
            .urdu {
                font-family: 'Noto Nastaliq Urdu', serif !important;
                direction: rtl !important;
                text-align: right !important;
                line-height: 2.05;
                letter-spacing: 0;
            }
            .field-label-ur {
                color: #4B5563;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
            }
            .field-wrap {
                margin: 0 0 20px 0;
            }
            .info-row {
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 12px;
            }
            .info-label {
                font-size: 13px;
                color: #6B7280;
                margin-bottom: 4px;
                font-weight: 600;
            }
            .info-value {
                font-size: 24px;
                color: #111827;
                font-weight: 700;
            }
            .info-note {
                font-size: 13px;
                color: #6B7280;
                line-height: 1.5;
            }
            .contacts-title {
                font-size: 20px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 10px;
            }
            .contacts-title-ur {
                font-size: 18px;
                color: #111827;
                margin-bottom: 12px;
            }
            .contacts-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }
            .contact-card {
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 12px;
            }
            .contact-label {
                font-size: 13px;
                color: #6B7280;
            }
            .contact-value {
                font-size: 22px;
                font-weight: 800;
                color: #111827;
            }
            .contact-detail {
                font-size: 13px;
                color: #4B5563;
                margin-top: 2px;
            }
            .contacts-note {
                font-size: 13px;
                color: #6B7280;
                margin-top: 10px;
            }
            .contacts-note-ur {
                font-size: 14px;
                color: #6B7280;
            }
            .action-panel {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 14px;
                padding: 14px;
                margin-bottom: 14px;
            }
            .action-title {
                font-size: 13px;
                font-weight: 700;
                color: #4B5563;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 10px;
            }
            .action-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 8px;
            }
            .action-tile {
                background: var(--tile-bg);
                border: 2px solid var(--tile-border);
                border-radius: 12px;
                padding: 12px 10px;
                text-align: center;
                opacity: var(--tile-opacity);
            }
            .action-icon {
                font-size: 18px;
                font-weight: 800;
                color: var(--tile-text);
            }
            .action-label {
                font-size: 16px;
                font-weight: 800;
                color: var(--tile-text);
            }
            .action-label-ur {
                font-size: 15px;
                font-weight: 700;
                color: var(--tile-text);
                font-family: 'Noto Nastaliq Urdu', serif;
                direction: rtl;
            }
            .action-detail {
                font-size: 17px;
                font-weight: 700;
                color: var(--action-text);
                margin-top: 12px;
            }
            .action-detail-ur {
                font-size: 16px;
                font-weight: 700;
                color: var(--action-text);
                font-family: 'Noto Nastaliq Urdu', serif;
                direction: rtl;
                text-align: right;
            }
            @media (max-width: 640px) {
                .contacts-grid {
                    grid-template-columns: 1fr;
                }
                .action-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            div[data-testid="stFormSubmitButton"] button,
            div.stButton > button {
                min-height: 52px;
                border-radius: 8px;
                border: 1px solid #1A56DB;
                background: #1A56DB;
                color: #FFFFFF;
                font-size: 18px;
                font-weight: 700;
                white-space: pre-line;
            }
            div[data-testid="stFormSubmitButton"] button:hover,
            div.stButton > button:hover {
                border-color: #1649BD;
                background: #1649BD;
                color: #FFFFFF;
            }
            .alert-banner {
                background: #FEF2F2;
                border: 1px solid #FCA5A5;
                border-left: 4px solid #DC2626;
                border-radius: 8px;
                padding: 18px;
                margin-bottom: 18px;
            }
            .alert-title {
                color: #7F1D1D;
                font-size: 18px;
                line-height: 1.45;
                font-weight: 800;
                margin: 0 0 12px 0;
            }
            .alert-item {
                border-top: 1px solid #FECACA;
                padding-top: 12px;
                margin-top: 12px;
            }
            .alert-line-en-summary {
                color: #7F1D1D;
                font-size: 17px;
                font-weight: 700;
                line-height: 1.4;
            }
            .alert-line-en-detail {
                color: #991B1B;
                font-size: 15px;
                font-weight: 600;
                line-height: 1.5;
                margin-top: 4px;
            }
            .alert-line-ur-label {
                color: #7F1D1D;
                font-size: 18px;
                font-weight: 700;
                line-height: 2;
                margin-top: 6px;
            }
            .alert-line-ur-detail {
                color: #991B1B;
                font-size: 16px;
                font-weight: 600;
                line-height: 2.05;
                margin-top: 2px;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_local_bundle():
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(ttl=60, show_spinner=False)
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
        backend_districts = sorted(_backend_get("/districts").get("districts", []))
        if backend_districts:
            return backend_districts
    except Exception:
        pass
    try:
        _, meta = _load_local_bundle()
        local_districts = sorted(meta.get("allowed_districts", []))
        if local_districts:
            return local_districts
    except Exception:
        pass
    return sorted(DISTRICT_RISK_PROFILE.keys())


def _feature_list() -> list[str]:
    try:
        return _backend_get("/features").get("features_used", FEATURES_USED)
    except Exception:
        return FEATURES_USED


@st.cache_data(ttl=60, show_spinner=False)
def _predict(payload: dict) -> dict:
    try:
        return _backend_predict(payload)
    except Exception:
        model, meta = _load_local_bundle()
        return predict_from_user_inputs(
            model=model,
            meta=meta,
            rainfall_mm=payload["rainfall_mm"],
            selected_date=date.fromisoformat(payload["selected_date"]),
            district=payload["district"],
            soil_condition=payload["soil_condition"],
            visible_water=payload["visible_water"],
        )


@st.cache_data(ttl=60, show_spinner=False)
def _compute_provincial_alerts(
    districts: list[str],
    rainfall_mm: float,
    selected_date: date,
    soil_condition: str,
    visible_water: str,
    exclude_district: str | None = None,
) -> list[dict]:
    alerts: list[dict] = []
    for district in districts:
        if exclude_district is not None and district == exclude_district:
            continue
        result = _predict(
            {
                "rainfall_mm": float(rainfall_mm),
                "selected_date": selected_date.isoformat(),
                "district": district,
                "soil_condition": soil_condition,
                "visible_water": visible_water,
            }
        )
        risk = _normalize_risk_level(result.get("risk_level_en"))
        if not result.get("ok", False) or risk not in {"High", "Critical"}:
            continue
        alerts.append(
            {
                "district": district,
                "risk_level_en": risk,
                "risk_level_ur": (result.get("risk_level_ur") or "").strip(),
                "recommended_action_en": (result.get("recommended_action_en") or "").strip(),
                "recommended_action_ur": (result.get("recommended_action_ur") or "").strip(),
            }
        )
    return alerts


def _render_header() -> None:
    st.markdown(
        """
        <div class="main-card">
          <h1 class="header-title">FloodSense</h1>
          <div class="header-title urdu">سیلاب کا خطرہ</div>
          <p class="header-subtitle">Enter today's field conditions to check flood risk for your district.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alert_banner(alert_groups: list[dict], *, show_exclude_hint: bool) -> None:
    if not alert_groups:
        return
    # Fragment strings must start at column 0: Streamlit Markdown treats lines
    # with leading spaces as indented code fences, which shows raw HTML as text.
    parts: list[str] = [
        '<div class="alert-banner">',
        '<p class="alert-title">High Risk Alert</p>',
        '<div class="alert-title urdu">خطرے کی انتباہ</div>',
    ]
    if show_exclude_hint:
        parts.append(
            '<p class="alert-banner-context" '
            'style="color:#991B1B;font-size:14px;line-height:1.45;font-weight:600;'
            'margin:4px 0 12px 0;">'
            "Additional regions flagged for these same rainfall and field inputs:"
            '</p>'
            '<p class="urdu alert-banner-context" '
            'style="color:#991B1B;font-size:15px;line-height:2;font-weight:600;'
            'margin:4px 0 12px 0;">انھی بارش اور میدانی حالات پر مزید علاقے زیادہ خطرے میں۔</p>'
        )
    for row in alert_groups:
        risk = row["risk_level_en"]
        disp = escape(
            ", ".join(_district_display_name(district_key) for district_key in row["districts"]),
        )
        risk_e = escape(risk)
        action_en = escape(row["recommended_action_en"])
        risk_ur = escape(row.get("risk_level_ur") or URDU_RISK_LABELS.get(risk, risk))
        action_ur = escape(row.get("recommended_action_ur") or URDU_ACTIONS.get(risk, ""))
        parts.extend(
            (
                '<div class="alert-item">',
                f'<div class="alert-line-en-summary">{disp}: {risk_e}.</div>',
                f'<div class="alert-line-en-detail">{action_en}</div>',
                f'<div class="urdu alert-line-ur-label">{risk_ur}</div>',
                f'<div class="urdu alert-line-ur-detail">{action_ur}</div>',
                "</div>",
            )
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _field_label(english: str, urdu: str) -> None:
    st.markdown(
        f"""
        <div class="field-wrap">
          <div class="field-label-en">{english}</div>
          <div class="field-label-ur urdu">{urdu}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _soil_display(value: str) -> str:
    return {"Dry": "Dry / خشک", "Moist": "Moist / نم", "Saturated": "Saturated / سیراب"}[value]


def _water_display(value: str) -> str:
    return {"Yes": "Yes / ہاں", "No": "No / نہیں"}[value]


def _render_emergency_contacts_card() -> None:
    rows = "".join(
        (
            '<div class="contact-card">'
            f'<div class="contact-label">{escape(item["label"])}</div>'
            f'<div class="contact-value">{escape(item["value"])}</div>'
            f'<div class="contact-detail">{escape(item["detail"])}</div>'
            "</div>"
        )
        for item in EMERGENCY_CONTACTS
    )
    st.markdown(
        '<div class="main-card">'
        '<div class="contacts-title">Emergency Contacts</div>'
        '<div class="urdu contacts-title-ur">ہنگامی رابطے</div>'
        f'<div class="contacts-grid">{rows}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_action_level_strip(risk: str) -> None:
    current = ACTION_LEVELS.get(risk, ACTION_LEVELS["Low"])
    items = []
    for level in ["Low", "Medium", "High", "Critical"]:
        style = RISK_STYLES[level]
        data = ACTION_LEVELS[level]
        active = level == risk
        items.append(
            '<div class="action-tile" '
            f'style="--tile-bg:{style["bg"] if active else "#FFFFFF"};'
            f'--tile-border:{style["border"] if active else "#E5E7EB"};'
            f'--tile-text:{style["text"]};'
            f'--tile-opacity:{"1" if active else "0.45"};">'
            f'<div class="action-icon">{escape(data["icon"])}</div>'
            f'<div class="action-label">{escape(data["label"])}</div>'
            f'<div class="action-label-ur">{escape(data["urdu"])}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="action-panel" '
        f'style="--action-text:{RISK_STYLES[risk]["text"]};">'
        '<div class="action-title">Current action level</div>'
        f'<div class="action-grid">{"".join(items)}</div>'
        f'<div class="action-detail">{escape(current["detail"])}</div>'
        f'<div class="action-detail-ur">{escape(current["urdu"])}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_audio_icon(risk: str) -> None:
    urdu_message = URDU_TTS_MESSAGES.get(risk, URDU_TTS_MESSAGES["Low"])
    components.html(
        f"""
        <div style="display:flex; align-items:center; justify-content:center; min-height:94px; font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
          <button id="floodsense-tts-btn" type="button" aria-label="Play Urdu audio summary" title="Play Urdu audio summary" style="width:62px; height:62px; border:1px solid #BFDBFE; border-radius:50%; background:linear-gradient(145deg,#EFF6FF,#DBEAFE); color:#1D4ED8; box-shadow:0 10px 22px rgba(29,78,216,0.18); display:grid; place-items:center; cursor:pointer;">
            <svg id="floodsense-tts-icon" width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 9.5V14.5H8L13 19V5L8 9.5H4Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
              <path d="M16 9C16.8 9.8 17.25 10.85 17.25 12C17.25 13.15 16.8 14.2 16 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              <path d="M18.5 6.5C20 8 20.75 9.85 20.75 12C20.75 14.15 20 16 18.5 17.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </button>
        </div>
        <script>
          (() => {{
            const button = document.getElementById("floodsense-tts-btn");
            const icon = document.getElementById("floodsense-tts-icon");
            const utteranceText = {json.dumps(urdu_message)};
            if (!window.speechSynthesis || typeof window.SpeechSynthesisUtterance === "undefined") {{
              button.disabled = true;
              button.style.opacity = "0.65";
              button.style.cursor = "not-allowed";
              button.title = "Speech playback is not available in this browser.";
              return;
            }}
            const resetButton = () => {{
              button.disabled = false;
              button.style.transform = "scale(1)";
              icon.style.opacity = "1";
            }};
            button.addEventListener("click", () => {{
              window.speechSynthesis.cancel();
              const utterance = new SpeechSynthesisUtterance(utteranceText);
              utterance.lang = "ur-PK";
              utterance.rate = 0.92;
              utterance.pitch = 1;
              utterance.volume = 1;
              button.disabled = true;
              button.style.transform = "scale(0.96)";
              icon.style.opacity = "0.55";
              utterance.onend = resetButton;
              utterance.onerror = () => {{
                window.speechSynthesis.cancel();
                resetButton();
              }};
              window.speechSynthesis.speak(utterance);
            }});
            window.addEventListener("beforeunload", () => window.speechSynthesis.cancel());
          }})();
        </script>
        """,
        height=102,
    )


def _store_result(rainfall_mm: float, selected_date: date, district: str, soil_condition: str, visible_water: str) -> None:
    payload = {
        "rainfall_mm": float(rainfall_mm),
        "selected_date": selected_date.isoformat(),
        "district": district,
        "soil_condition": soil_condition,
        "visible_water": visible_water,
    }
    st.session_state["latest_payload"] = payload
    st.session_state["latest_result"] = _predict(payload)
    st.session_state["show_result"] = True


def _render_input_card(districts: list[str]) -> None:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Field Conditions</h2>', unsafe_allow_html=True)
    with st.form("field_conditions_form", clear_on_submit=False):
        _field_label("How much rain fell today?", "آج کتنی بارش ہوئی؟")
        rainfall_mm = st.number_input("Rainfall in millimetres", min_value=0, max_value=500, value=int(st.session_state.get("rainfall_input", 0) or 0), step=1, key="rainfall_input", label_visibility="collapsed")
        _field_label("Today's date", "آج کی تاریخ")
        selected_date = st.date_input("Today's date", value=st.session_state.get("date_input", date.today()), key="date_input", label_visibility="collapsed")
        _field_label("District", "ضلع")
        district = st.selectbox("District", options=districts, key="district_input", label_visibility="collapsed")
        _field_label("How wet is the soil?", "مٹی کتنی گیلی ہے؟")
        soil_condition = st.radio("How wet is the soil?", options=["Dry", "Moist", "Saturated"], index=["Dry", "Moist", "Saturated"].index(st.session_state.get("soil_input", "Moist")), key="soil_input", format_func=_soil_display, horizontal=True, label_visibility="collapsed")
        _field_label("Is there standing water visible?", "کیا پانی جمع نظر آتا ہے؟")
        visible_water = st.radio("Is there standing water visible?", options=["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("water_input", "No")), key="water_input", format_func=_water_display, horizontal=True, label_visibility="collapsed")
        run = st.form_submit_button("Check Flood Risk\nسیلاب کا خطرہ جانچیں", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if run:
        _store_result(rainfall_mm, selected_date, district, soil_condition, visible_water)
        st.rerun()
    _render_emergency_contacts_card()


def _render_warning_result(message: str) -> None:
    st.markdown(
        """
        <div style="background:#FFFBEB; border:1px solid #FCD34D; border-radius:12px; padding:20px; text-align:center;">
          <div style="font-size:20px; font-weight:600; color:#92400E;">Insufficient data</div>
          <div style="font-size:15px; color:#78350F; margin-top:6px;">Please check your inputs and try again.</div>
          <div style="font-size:15px; color:#78350F; font-family:'Noto Nastaliq Urdu', serif; direction:rtl; margin-top:8px;">ڈیٹا ناکافی ہے — براہ کرم اپنی معلومات جانچیں۔</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _scroll_page_to_top() -> None:
        components.html(
                """
                <script>
                    (function () {
                        const scrollTop = () => {
                            try {
                                const parentDocument = window.parent.document;
                                parentDocument.documentElement.scrollTop = 0;
                                parentDocument.body.scrollTop = 0;
                                window.parent.scrollTo(0, 0);
                            } catch (error) {
                                try {
                                    window.top.scrollTo(0, 0);
                                } catch (fallbackError) {
                                    window.scrollTo(0, 0);
                                }
                            }
                        };
                        requestAnimationFrame(() => requestAnimationFrame(scrollTop));
                    })();
                </script>
                """,
                height=1,
        )


def _render_output_card(result: dict) -> None:
    if not result.get("ok", False):
        _render_warning_result(result.get("message", INSUFFICIENT_DATA_MESSAGE))
        return

    risk = result["risk_level_en"]
    badge = RISK_STYLES.get(risk, RISK_STYLES["Low"])
    risk_ur = URDU_RISK_LABELS.get(risk, result.get("risk_level_ur", ""))
    action_ur = URDU_ACTIONS.get(risk, result.get("recommended_action_ur", ""))
    action_border = "#1A56DB"
    action_bg = "#EFF6FF"
    if risk == "Critical":
        action_border = "#DC2626"
        action_bg = "#FFF1F2"
    elif risk == "High":
        action_border = "#EA580C"
        action_bg = "#FFF7ED"

    st.markdown(
        f"""
        <div style="background:{badge['bg']}; border:2px solid {badge['border']}; border-radius:16px; padding:28px 24px; text-align:center; margin-bottom:20px;">
          <div style="font-size:42px; font-weight:700; color:{badge['text']}; line-height:1.1;">{risk}</div>
          <div style="font-size:40px; font-weight:700; color:{badge['text']}; font-family:'Noto Nastaliq Urdu', serif; direction:rtl; margin-top:8px;">{risk_ur}</div>
        </div>
        <div class="info-row">
          <div class="info-label">Confidence - How certain the system is</div>
          <div class="urdu" style="font-size:13px; color:#6B7280;">اعتماد — نظام کتنا یقین رکھتا ہے</div>
          <div class="info-value">{result["confidence_pct"]:.1f}%</div>
        </div>
        <div class="info-row">
          <div class="info-label">People estimated at risk</div>
          <div class="urdu" style="font-size:13px; color:#6B7280;">خطرے میں متوقع آبادی</div>
          <div class="info-value">{result["population_risk_estimate"]:,}</div>
        </div>
        <div style="background:{action_bg}; border-left:4px solid {action_border}; border-radius:0 12px 12px 0; padding:16px 20px; margin-bottom:12px;">
          <div style="font-size:13px; font-weight:600; color:#1E40AF; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;">Action Required / کارروائی درکار ہے</div>
          <div style="font-size:17px; color:#111827; line-height:1.6;">{result["recommended_action_en"]}</div>
          <div style="font-size:20px; color:#1E3A5F; font-family:'Noto Nastaliq Urdu', serif; direction:rtl; text-align:right; margin-top:10px; line-height:2;">{action_ur}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, audio_col = st.columns([5, 1], gap="small")
    with audio_col:
        _render_audio_icon(risk)
    _render_action_level_strip(risk)
    _render_emergency_contacts_card()


def main() -> None:
    _inject_styles()
    _render_header()

    if not ARTIFACTS_DIR.exists():
        st.error("The flood check is not ready yet. Please run setup first.")
        return

    districts = _district_options()
    if not districts:
        st.error("No districts are available yet.")
        return

    if st.session_state.get("show_result", False):
        focus = (st.session_state.get("latest_payload") or {}).get("district") or ""
        alerts = _compute_provincial_alerts(
            districts=districts,
            rainfall_mm=float(st.session_state.get("rainfall_input", 0.0)),
            selected_date=st.session_state.get("date_input", date.today()),
            soil_condition=st.session_state.get("soil_input", "Moist"),
            visible_water=st.session_state.get("water_input", "No"),
            exclude_district=focus or None,
        )
        alert_groups = merge_provincial_alerts_by_same_advisory(alerts)
        _render_alert_banner(alert_groups, show_exclude_hint=bool(focus))
        if st.button("Back / واپس", key="back_top", use_container_width=False):
            st.session_state["show_result"] = False
            st.rerun()
        _scroll_page_to_top()
        _render_output_card(st.session_state.get("latest_result", {"ok": False, "message": INSUFFICIENT_DATA_MESSAGE}))
        return

    _render_input_card(districts)


if __name__ == "__main__":
    main()
