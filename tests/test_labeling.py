import pandas as pd

from src.labels.relabel_time_windows import assign_label
from src.models.sensitivity import label_relative_minutes, shifted_relative_minutes


def test_relabel_time_window_boundaries_match_config():
    cfg = {
        "normal_range": {"start": -480, "end": -255},
        "positive_range": {"start": -225, "end": -15},
        "ignore_ranges": [[-255, -225], [-15, float("inf")]],
    }
    assert assign_label(-300, cfg) == 0
    assert assign_label(-240, cfg) == -1
    assert assign_label(-120, cfg) == 1
    assert assign_label(-5, cfg) == -1


def test_labeling_boundary_points_are_consistent():
    cfg = {
        "normal_range": {"start": -480, "end": -255},
        "positive_range": {"start": -225, "end": -15},
        "ignore_ranges": [[-255, -225], [-15, float("inf")]],
    }

    assert assign_label(-480, cfg) == 0
    assert assign_label(-255, cfg) == -1
    assert assign_label(-225, cfg) == 1
    assert assign_label(-15, cfg) == -1


def test_sensitivity_labeling_uses_horizon_and_shift():
    assert label_relative_minutes(-300, horizon_minutes=240) == 0
    assert label_relative_minutes(-240, horizon_minutes=240) == -1
    assert label_relative_minutes(-225, horizon_minutes=240) == 1
    assert label_relative_minutes(-120, horizon_minutes=240) == 1
    assert label_relative_minutes(-15, horizon_minutes=240) == -1

    shifted = shifted_relative_minutes(pd.Series([-120]), 15)
    assert float(shifted.iloc[0]) == -135.0
