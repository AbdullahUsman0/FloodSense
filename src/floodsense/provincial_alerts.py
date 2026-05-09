from __future__ import annotations


_SEVERITY = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def merge_provincial_alerts_by_same_advisory(rows: list[dict]) -> list[dict]:
    """
    Collapse provinces that carry the same risk band and bilingual advisory so the
    UI does not repeat identical paragraphs once per district.
    """
    if not rows:
        return []

    buckets: dict[tuple[str, str, str, str], list[str]] = {}
    for row in rows:
        key = (
            row["risk_level_en"],
            (row.get("risk_level_ur") or "").strip(),
            (row.get("recommended_action_en") or "").strip(),
            (row.get("recommended_action_ur") or "").strip(),
        )
        buckets.setdefault(key, []).append(row["district"])

    merged: list[dict] = []
    for key, district_keys in buckets.items():
        risk_en, risk_ur, act_en, act_ur = key
        merged.append(
            {
                "districts": sorted(district_keys),
                "risk_level_en": risk_en,
                "risk_level_ur": risk_ur,
                "recommended_action_en": act_en,
                "recommended_action_ur": act_ur,
            }
        )

    merged.sort(
        key=lambda r: (_SEVERITY.get(r["risk_level_en"], 99), r["districts"][0].lower()),
    )
    return merged
