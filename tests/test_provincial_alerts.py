"""Unit tests for provincial alert grouping (unittest discovery compatible)."""

import unittest

from floodsense.provincial_alerts import merge_provincial_alerts_by_same_advisory


def _base_row(name: str) -> dict:
    return {
        "district": name,
        "risk_level_en": "High",
        "risk_level_ur": "زیادہ",
        "recommended_action_en": "Issue district warning.",
        "recommended_action_ur": "ضلعی وارننگ جاری کریں۔",
    }


class TestMergeProvincialAlerts(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(merge_provincial_alerts_by_same_advisory([]), [])

    def test_single_row_round_trip(self) -> None:
        rows = [_base_row("Sindh_District")]
        merged = merge_provincial_alerts_by_same_advisory(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["districts"], ["Sindh_District"])

    def test_merges_identical_actions(self) -> None:
        a = _base_row("Sindh_District")
        b = _base_row("KP_District")
        c = _base_row("Jacobabad")
        merged = merge_provincial_alerts_by_same_advisory([a, b, c])
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]["districts"]), {"Jacobabad", "KP_District", "Sindh_District"})

    def test_splits_different_critical_actions(self) -> None:
        r1 = {**_base_row("Sindh_District"), "risk_level_en": "Critical", "risk_level_ur": "انتہائی"}
        r2 = {
            **_base_row("KP_District"),
            "risk_level_en": "Critical",
            "risk_level_ur": "انتہائی",
            "recommended_action_en": "Evacuate now.",
            "recommended_action_ur": "ابھی انخلا کریں۔",
        }
        merged = merge_provincial_alerts_by_same_advisory([r1, r2])
        self.assertEqual(len(merged), 2)

    def test_critical_orders_before_high(self) -> None:
        high_row = _base_row("Jacobabad")
        crit_row = {
            **_base_row("Sindh_District"),
            "risk_level_en": "Critical",
            "risk_level_ur": "انتہائی",
            "recommended_action_en": "Different action only for critical tier.",
            "recommended_action_ur": "منتقلی مختلف۔",
        }
        merged = merge_provincial_alerts_by_same_advisory([high_row, crit_row])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["risk_level_en"], "Critical")
        self.assertEqual(merged[1]["risk_level_en"], "High")

    def test_whitespace_stripped_so_same_advisory_merges(self) -> None:
        a = _base_row("A")
        b = _base_row("B")
        b["recommended_action_en"] = "  Issue district warning.  "
        merged = merge_provincial_alerts_by_same_advisory([a, b])
        self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()
