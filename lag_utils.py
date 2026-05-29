"""
Half / overtime-aware momentum lag features.

In CS:GO the two teams swap sides at the half (round 15 -> 16) and again every
three rounds in overtime. ``ct_wins_round`` is a SIDE outcome, not a team
outcome, so a naive within-match ``shift()`` makes the momentum lag at the first
round of each new side refer to the *other* team's previous result. That injects
a mislabelled momentum signal at every half/OT boundary (most visibly round 16).

This helper recomputes the lags within (match_id, side-segment) blocks, leaving
the lag undefined (NaN) for the first round of each segment -- exactly the way
the very first round of a match is already excluded. Analysis scripts then drop
NaN-lag rows as before, so the contaminated boundary rounds are excluded rather
than mislabelled.

It is safe to call on a dataframe that already has ``ct_won_lag_*`` columns; the
columns are overwritten from ``ct_wins_round`` + ``round_num`` (both unaffected
by the bug), so results depend only on tracked code and not on any cached lag.
"""

from typing import Union

import pandas as pd

# Columns required by add_halfaware_lags.
_REQUIRED_COLUMNS = {'match_id', 'round_num', 'ct_wins_round'}

# Internal grouping column; suffixed to avoid clobbering a caller's column.
_SEGMENT_COL = '_side_seg_halfaware_tmp'


def side_segment(round_num: Union[int, float]) -> Union[int, float]:
    """Id for the contiguous block during which a team stays on one side.

    0 = first half (rounds 1-15), 1 = second half (16-30),
    2, 3, ... = successive 3-round overtime segments (31-33, 34-36, ...).

    Accepts floats because pandas ``Series.map`` passes values as float64.
    Returns NaN for NaN input (e.g. the synthetic "next round" of a match's
    last round), so callers can compare segments without special-casing.
    """
    if pd.isna(round_num):
        return float('nan')
    round_num = int(round_num)
    if round_num <= 15:
        return 0
    if round_num <= 30:
        return 1
    return 2 + (round_num - 31) // 3


def add_halfaware_lags(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """(Re)compute ``ct_won_lag_1..max_lag`` without crossing side switches."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"add_halfaware_lags: missing required columns {sorted(missing)}")

    df = df.sort_values(['match_id', 'round_num']).reset_index(drop=True)
    df[_SEGMENT_COL] = df['round_num'].map(side_segment)
    grouped = df.groupby(['match_id', _SEGMENT_COL])['ct_wins_round']
    for lag in range(1, max_lag + 1):
        df[f'ct_won_lag_{lag}'] = grouped.shift(lag)
    return df.drop(columns=_SEGMENT_COL)
