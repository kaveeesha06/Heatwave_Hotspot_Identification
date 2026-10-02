"""
Module: features.py
Description: Per-grid-cell feature engineering describing heat behavior across India.
Deliberately excludes latitude and longitude from feature sets so clusters reflect thermal behavior,
not geographic proximity.
Aligned with AI Use Case KJS-CES-01: Climate Intelligence for Heatwave Monitoring.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.load_data import load_heatwave_season_dataframe, get_available_raw_years



# Core behavioral features used for K-Means clustering (Lat & Lon strictly excluded)
CORE_HEAT_FEATURES = [
    "tmax_mean",
    "tmax_max",
    "hot_days_ge_40",
    "heatwave_days",
    "longest_hot_streak",
    "tmax_std"
]

EXTENDED_HEAT_FEATURES = CORE_HEAT_FEATURES + [
    "severe_heatwave_days",
    "tmax_p90"
]


def calculate_longest_hot_streak(temperatures: Union[np.ndarray, pd.Series], threshold: float = 40.0) -> int:
    """
    Computes the maximum length of consecutive days where Tmax >= threshold.
    """
    arr = np.asarray(temperatures)
    # Mask NaNs as False
    hot_mask = (~np.isnan(arr)) & (arr >= threshold)
    if not np.any(hot_mask):
        return 0
    
    # Vectorized streak length computation
    max_streak = 0
    current_streak = 0
    for val in hot_mask:
        if val:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def compute_grid_features(
    df_season: pd.DataFrame,
    hot_threshold: float = 40.0,
    severe_threshold: float = 45.0
) -> pd.DataFrame:
    """
    Computes per-grid-cell seasonal heat features.
    
    Features engineered per grid cell:
      - tmax_mean: Mean seasonal Tmax (°C)
      - tmax_max: Maximum Tmax recorded (°C)
      - hot_days_ge_40: Number of days with Tmax >= hot_threshold (default 40°C)
      - heatwave_days: Days with Tmax >= 45°C or (Tmax >= 40°C and Departure >= +4.5°C)
      - severe_heatwave_days: Days with Tmax >= 47°C or (Tmax >= 40°C and Departure >= +6.4°C)
      - longest_hot_streak: Longest consecutive hot-day streak (>= 40°C)
      - tmax_std: Standard deviation of seasonal Tmax (°C)
      - tmax_p90: 90th percentile of Tmax (°C)
      - total_days: Count of valid observation days
    """
    # Sort by grid_id and date for streak calculation
    df_sorted = df_season.sort_values(["grid_id", "date"]).copy()

    # Pre-calculate cell climatological baseline normal (mean across all years for that cell)
    cell_normals = df_sorted.groupby("grid_id")["tmax"].transform("mean")
    df_sorted["tmax_anomaly"] = df_sorted["tmax"] - cell_normals

    # Heatwave day flags per IMD criteria
    # IMD criteria: Tmax >= 40°C and departure >= 4.5°C, or severe threshold Tmax >= 45°C
    df_sorted["is_hot_day"] = df_sorted["tmax"] >= hot_threshold
    df_sorted["is_heatwave"] = (df_sorted["tmax"] >= severe_threshold) | (
        (df_sorted["tmax"] >= hot_threshold) & (df_sorted["tmax_anomaly"] >= 4.5)
    )
    df_sorted["is_severe_heatwave"] = (df_sorted["tmax"] >= 47.0) | (
        (df_sorted["tmax"] >= hot_threshold) & (df_sorted["tmax_anomaly"] >= 6.4)
    )

    # Group aggregations
    features = []
    for grid_id, group in df_sorted.groupby("grid_id"):
        tvals = group["tmax"].dropna().values
        if len(tvals) == 0:
            continue

        lat = group["lat"].iloc[0]
        lon = group["lon"].iloc[0]
        region = group["region"].iloc[0]

        mean_t = float(np.mean(tvals))
        max_t = float(np.max(tvals))
        std_t = float(np.std(tvals))
        p90_t = float(np.percentile(tvals, 90))

        hot_days = int(group["is_hot_day"].sum())
        hw_days = int(group["is_heatwave"].sum())
        sev_hw_days = int(group["is_severe_heatwave"].sum())
        streak = calculate_longest_hot_streak(group["tmax"].values, threshold=hot_threshold)
        n_days = int(len(tvals))

        features.append({
            "grid_id": grid_id,
            "lat": lat,
            "lon": lon,
            "region": region,
            "tmax_mean": round(mean_t, 2),
            "tmax_max": round(max_t, 2),
            "hot_days_ge_40": hot_days,
            "heatwave_days": hw_days,
            "severe_heatwave_days": sev_hw_days,
            "longest_hot_streak": streak,
            "tmax_std": round(std_t, 2),
            "tmax_p90": round(p90_t, 2),
            "total_days": n_days
        })

    feat_df = pd.DataFrame(features)
    return feat_df


def build_and_save_features(
    raw_data_dir: Union[str, Path] = "data/raw",
    processed_dir: Union[str, Path] = "data/processed",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads raw IMD data, computes aggregate multi-year and individual year features,
    and caches them as Parquet files in data/processed/.
    """
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    avail_years = get_available_raw_years(raw_data_dir)
    if not avail_years:
        print("No raw data found in data/raw. Will trigger download or synthetic generation...")
        avail_years = [2022, 2023, 2024]

    s_yr = start_year or min(avail_years)
    e_yr = end_year or max(avail_years)

    print(f"Loading heatwave season data (March to June) for years {s_yr} to {e_yr}...")
    df_season = load_heatwave_season_dataframe(
        start_year=s_yr,
        end_year=e_yr,
        data_dir=raw_data_dir,
        season_months=(3, 4, 5, 6)
    )

    # 1. Multi-year aggregate features
    print("Computing multi-year aggregate features...")
    agg_features = compute_grid_features(df_season)
    agg_parquet_path = proc_path / "features_heatwave_season.parquet"
    agg_features.to_parquet(agg_parquet_path, index=False)
    print(f"Saved aggregate features to {agg_parquet_path} ({len(agg_features)} grid cells)")

    # 2. Individual year features
    year_dfs = {}
    for yr in range(s_yr, e_yr + 1):
        df_yr = df_season[df_season["year"] == yr]
        if len(df_yr) > 0:
            print(f"Computing features for year {yr}...")
            yr_feat = compute_grid_features(df_yr)
            yr_path = proc_path / f"features_{yr}.parquet"
            yr_feat.to_parquet(yr_path, index=False)
            year_dfs[yr] = yr_feat
            print(f"Saved {yr} features to {yr_path}")

    return agg_features, year_dfs


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Feature Engineering Pipeline (src/features.py)")
    print("=" * 60)
    agg_feats, yr_feats = build_and_save_features()
    print("\nFeature Summary Statistics (Multi-year aggregate):")
    print(agg_feats[CORE_HEAT_FEATURES].describe().round(2))
