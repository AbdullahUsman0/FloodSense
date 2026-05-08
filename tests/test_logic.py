from datetime import date

from floodsense.inference import (
    apply_operational_risk_guardrail,
    estimate_population_at_risk,
    risk_from_probability,
    validate_user_inputs,
)


def test_risk_band_boundaries():
    assert risk_from_probability(0.00)[0] == "Low"
    assert risk_from_probability(0.2499)[0] == "Low"
    assert risk_from_probability(0.25)[0] == "Medium"
    assert risk_from_probability(0.50)[0] == "Medium"
    assert risk_from_probability(0.5001)[0] == "High"
    assert risk_from_probability(0.75)[0] == "High"
    assert risk_from_probability(0.7501)[0] == "Critical"


def test_input_validation_guardrails():
    allowed = {"Sindh_District", "KP_District"}
    valid, _ = validate_user_inputs(55.0, date(2026, 8, 1), "Sindh_District", "Moist", "Yes", allowed)
    assert valid

    invalid, msg = validate_user_inputs(None, date(2026, 8, 1), "Sindh_District", "Moist", "Yes", allowed)
    assert not invalid
    assert "Insufficient data" in msg

    invalid, _ = validate_user_inputs(900.0, date(2026, 8, 1), "Sindh_District", "Moist", "Yes", allowed)
    assert not invalid


def test_extreme_rain_guardrail_escalates_to_critical():
    level = apply_operational_risk_guardrail(
        base_risk_level_en="Medium",
        rainfall_mm=480.0,
        soil_condition="Saturated",
        visible_water="Yes",
    )
    assert level == "Critical"


def test_population_estimate_is_bounded():
    ndma = {"Sindh": 14_563_770}
    estimate = estimate_population_at_risk(
        district="Sindh_District",
        ndma_population_by_region=ndma,
        risk_level="Critical",
        confidence_pct=90.0,
    )
    # With divisor 2 for Sindh and profile caps, estimate should stay well below 1M.
    assert estimate < 1_000_000


def test_district_profile_weighting_is_more_conservative_for_sparse_areas():
    ndma = {"Sindh": 14_563_770, "Balochistan": 9_182_616}
    sindh_est = estimate_population_at_risk(
        district="Sindh_District",
        ndma_population_by_region=ndma,
        risk_level="Critical",
        confidence_pct=85.0,
    )
    bal_est = estimate_population_at_risk(
        district="Balochistan_District",
        ndma_population_by_region=ndma,
        risk_level="Critical",
        confidence_pct=85.0,
    )
    # Sparse plateau profile should scale down exposed counts relative to dense floodplain profiles.
    assert bal_est < sindh_est
