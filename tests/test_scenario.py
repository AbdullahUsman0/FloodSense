from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodsense.scenario import apply_scenario_card1_monsoon_surge
from floodsense.scenario import apply_scenario_card2_sensor_rogue


def test_scenario_card1_injects_two_rows():
    df = pd.read_csv(ROOT / "floodsense_training_data.csv")
    augmented, injected = apply_scenario_card1_monsoon_surge(df)
    assert len(injected) == 2
    assert len(augmented) == len(df) + 2
    impacted = {item.district for item in injected}
    assert impacted == {"Sindh_District", "KP_District"}


def test_scenario_card2_imputes_faulty_district_without_drop():
    df = pd.read_csv(ROOT / "floodsense_training_data.csv")
    updated, detail = apply_scenario_card2_sensor_rogue(df, faulty_district="Sindh_District")
    assert detail is not None
    assert detail.faulty_district == "Sindh_District"
    assert len(detail.donor_districts) == 2
    assert len(detail.donor_values) == 2
    assert len(updated) == len(df)
    assert updated["district"].eq("Sindh_District").any()
