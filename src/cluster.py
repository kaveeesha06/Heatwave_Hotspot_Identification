"""
Module: cluster.py
Description: Feature scaling, K-Means clustering, k-selection diagnostics (Elbow & Silhouette),
cluster ranking and severity labeling (Low, Moderate, High, Extreme Hotspot).
Aligned with AI Use Case KJS-CES-01: Climate Intelligence for Heatwave Monitoring.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.features import CORE_HEAT_FEATURES


# Canonical severity names based on number of clusters k
SEVERITY_MAPPINGS = {
    2: ["Non-Hotspot", "Heatwave Hotspot"],
    3: ["Low", "Moderate", "Extreme (Hotspot)"],
    4: ["Low", "Moderate", "High", "Extreme (Hotspot)"],
    5: ["Low", "Mild", "Moderate", "High", "Extreme (Hotspot)"],
    6: ["Very Low", "Low", "Moderate", "Elevated", "High", "Extreme (Hotspot)"],
    7: ["Minimal", "Very Low", "Low", "Moderate", "Elevated", "High", "Extreme (Hotspot)"]
}

# Color palette mapped to severity tiers for consistent visualizations
SEVERITY_COLORS = {
    "Low": "#2ca02c",                  # Green
    "Mild": "#8c564b",
    "Minimal": "#17becf",
    "Very Low": "#2ca02c",
    "Moderate": "#ffbb78",              # Light orange / peach
    "Elevated": "#ff7f0e",
    "High": "#d62728",                  # Red
    "Extreme (Hotspot)": "#7f0000",     # Deep maroon / crimson
    "Heatwave Hotspot": "#7f0000",
    "Non-Hotspot": "#2ca02c"
}


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Extracts numerical features and standardizes them using StandardScaler.
    Lat and Lon are excluded from clustering features.
    """
    cols = feature_cols or CORE_HEAT_FEATURES
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features missing from DataFrame: {missing_cols}")

    X_raw = df[cols].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    return X_scaled, scaler, cols


def evaluate_k_range(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Runs K-Means across k_min to k_max to compute:
      - Inertia (Elbow method: within-cluster sum of squares)
      - Silhouette Score (higher is better, range [-1, 1])
      - Davies-Bouldin Index (lower is better)
      - Calinski-Harabasz Score (higher is better)
    """
    X_scaled, _, _ = prepare_feature_matrix(df, feature_cols=feature_cols)

    diagnostics = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertia = float(km.inertia_)
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
        ch_score = float(calinski_harabasz_score(X_scaled, labels))

        diagnostics.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin": round(db_score, 4),
            "calinski_harabasz": round(ch_score, 2)
        })

    return pd.DataFrame(diagnostics)


def fit_kmeans_and_label(
    df: pd.DataFrame,
    k: int = 4,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Fits K-Means for a given k, ranks clusters monotonically by heat severity,
    assigns human-interpretable severity labels (Low, Moderate, High, Extreme Hotspot),
    and computes cluster profile summaries.
    """
    cols = feature_cols or CORE_HEAT_FEATURES
    X_scaled, scaler, _ = prepare_feature_matrix(df, feature_cols=cols)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(X_scaled)

    # Attach raw cluster IDs temporarily
    temp_df = df.copy()
    temp_df["_raw_cluster"] = raw_labels

    # Rank clusters by mean hot days (or mean seasonal Tmax) in ascending order
    # (Rank 0 = coolest cluster, Rank k-1 = hottest hotspot cluster)
    ranking_metric = "hot_days_ge_40" if "hot_days_ge_40" in temp_df.columns else "tmax_mean"
    cluster_means = temp_df.groupby("_raw_cluster")[ranking_metric].mean().sort_values()
    raw_to_ranked = {raw_id: rank for rank, raw_id in enumerate(cluster_means.index)}

    # Map rank to canonical labels
    label_names = SEVERITY_MAPPINGS.get(k, [f"Tier {i+1}" for i in range(k-1)] + ["Extreme (Hotspot)"])
    rank_to_label = {rank: label_names[rank] for rank in range(k)}

    temp_df["cluster_id"] = temp_df["_raw_cluster"].map(raw_to_ranked)
    temp_df["cluster_name"] = temp_df["cluster_id"].map(rank_to_label)
    
    top_cluster_id = k - 1
    temp_df["is_hotspot"] = temp_df["cluster_id"] == top_cluster_id
    temp_df.drop(columns=["_raw_cluster"], inplace=True)

    # Compute evaluation metrics
    sil = float(silhouette_score(X_scaled, temp_df["cluster_id"].values))
    db = float(davies_bouldin_score(X_scaled, temp_df["cluster_id"].values))
    inertia = float(kmeans.inertia_)

    # Generate Cluster Profiles Summary Table
    profile_aggs = {
        "grid_id": "count",
        "tmax_mean": "mean",
        "tmax_max": "max",
        "hot_days_ge_40": "mean",
        "heatwave_days": "mean",
        "longest_hot_streak": "mean",
        "tmax_std": "mean"
    }
    # Keep only available columns
    actual_aggs = {c: agg for c, agg in profile_aggs.items() if c in temp_df.columns}
    
    profiles = temp_df.groupby(["cluster_id", "cluster_name"]).agg(actual_aggs).reset_index()
    profiles.rename(columns={"grid_id": "cell_count"}, inplace=True)
    profiles["pct_land_area"] = (profiles["cell_count"] / len(temp_df) * 100).round(1)

    # Round numeric columns for display
    float_cols = profiles.select_dtypes(include=[np.floating]).columns
    profiles[float_cols] = profiles[float_cols].round(2)

    metadata = {
        "k": k,
        "features": cols,
        "silhouette_score": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "inertia": round(inertia, 2),
        "scaler": scaler,
        "kmeans_model": kmeans,
        "hotspot_cluster_name": rank_to_label[top_cluster_id]
    }

    return temp_df, profiles, metadata


def validate_against_known_heat_zones(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates cluster assignments against known heatwave-prone zones of India
    (e.g., Vidarbha, Rajasthan, Odisha, Coastal AP, Peninsular).
    Cross-tabulates region vs cluster severity.
    """
    cross_tab = pd.crosstab(
        labeled_df["region"],
        labeled_df["cluster_name"],
        margins=True,
        margins_name="Total"
    )
    return cross_tab


if __name__ == "__main__":
    from src.features import build_and_save_features

    proc_file = Path("data/processed/features_heatwave_season.parquet")
    if not proc_file.exists():
        df_feat, _ = build_and_save_features()
    else:
        df_feat = pd.read_parquet(proc_file)

    print("=" * 60)
    print("Evaluating K-Means Diagnostics (src/cluster.py)")
    print("=" * 60)
    diagnostics = evaluate_k_range(df_feat, k_min=2, k_max=8)
    print(diagnostics.to_string(index=False))

    print("\n" + "=" * 60)
    print("Fitting K-Means with k=4 (Low, Moderate, High, Extreme)")
    print("=" * 60)
    labeled_df, profiles, meta = fit_kmeans_and_label(df_feat, k=4)
    print(f"Silhouette Score: {meta['silhouette_score']} | Davies-Bouldin: {meta['davies_bouldin']}")
    print("\nCluster Profiles Summary Table:")
    print(profiles.to_string(index=False))

    print("\nValidation against Known Indian Meteorological Zones:")
    val_table = validate_against_known_heat_zones(labeled_df)
    print(val_table.to_string())
