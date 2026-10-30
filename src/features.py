"""
Feature engineering for classical-ML NILM ON/OFF classification.

Two arms, built to directly test whether the literature's 0.53-vs-0.98 spread
is about features rather than an inherent classical-ML ceiling:

  Arm A -- raw/minimal: replicates the weaker classical-ML studies
      (Tabatabaei et al. 2017, F1=0.528 on REDD).
  Arm B -- engineered: rolling statistics, deltas, distributional shape,
      time-of-day -- replicating what plausibly explains the stronger
      classical-ML studies (Rehmani et al. 2021, ~98% accuracy on REDD, though
      note that paper reports accuracy, not F1).

Window length is a stated, deliberate choice: DEFAULT_WINDOW=15 samples is
about 2 minutes at REFIT's ~8s sampling. Treated as a documented parameter,
not tuned per appliance in the core sweep (a window-length sensitivity check
is future work, not core scope).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_WINDOW = 15  # samples, ~2 minutes at REFIT's ~8s sampling


def arm_a_features(aggregate: pd.Series) -> pd.DataFrame:
    """Raw/minimal feature set."""
    out = pd.DataFrame(index=aggregate.index)
    out["power"] = aggregate.values
    if isinstance(aggregate.index, pd.DatetimeIndex):
        out["hour"] = aggregate.index.hour
    return out


def arm_b_features(aggregate: pd.Series, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Engineered feature set: rolling stats, deltas, distributional shape, time."""
    s = aggregate.astype(float)
    out = pd.DataFrame(index=s.index)
    out["power"] = s.values
    roll = s.rolling(window, min_periods=1)
    out["roll_mean"] = roll.mean().values
    out["roll_std"] = roll.std().fillna(0).values
    out["roll_min"] = roll.min().values
    out["roll_max"] = roll.max().values
    out["roll_skew"] = roll.skew().fillna(0).values
    out["roll_kurt"] = roll.kurt().fillna(0).values
    out["delta"] = s.diff().fillna(0).values
    out["delta_abs_mean"] = s.diff().abs().rolling(window, min_periods=1).mean().fillna(0).values
    if isinstance(s.index, pd.DatetimeIndex):
        out["hour"] = s.index.hour
        out["dow"] = s.index.dayofweek
    return out


def build_feature_matrix(aggregate: pd.Series, arm: str = "B",
                          window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    if arm.upper() == "A":
        return arm_a_features(aggregate)
    if arm.upper() == "B":
        return arm_b_features(aggregate, window=window)
    raise ValueError(f"Unknown arm: {arm!r}")
