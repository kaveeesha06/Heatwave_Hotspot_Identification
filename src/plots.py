"""
Module: plots.py
Description: Interactive diagnostic and geographical visualization functions for heatwave hotspot analytics.
Compatible with Plotly 5, 6, and 7+.
Aligned with AI Use Case KJS-CES-01: Climate Intelligence for Heatwave Monitoring.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.cluster import SEVERITY_COLORS


def plot_hotspot_map(
    labeled_df: pd.DataFrame,
    title: str = "IMD Gridded Heatwave Hotspot Identification Across India",
    zoom: float = 3.6,
    center_lat: float = 22.5,
    center_lon: float = 82.0,
    point_size: int = 14
) -> go.Figure:
    """
    Renders an interactive geographic scatter map of India with grid cells colored by cluster severity.
    Includes rich hover tooltips with thermal metrics. Compatible with Plotly 5, 6, and 7+.
    """
    df = labeled_df.copy()
    df = df.sort_values("cluster_id")

    hover_data = {
        "lat": ":.2f",
        "lon": ":.2f",
        "region": True,
        "cluster_name": True,
        "tmax_mean": ":.2f",
        "tmax_max": ":.2f",
        "hot_days_ge_40": True,
        "heatwave_days": True,
        "cluster_id": False
    }

    color_map = {name: SEVERITY_COLORS.get(name, "#333333") for name in df["cluster_name"].unique()}

    # Plotly 7+ uses px.scatter_map / go.Scattermap, Plotly <7 uses px.scatter_mapbox
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df,
            lat="lat",
            lon="lon",
            color="cluster_name",
            color_discrete_map=color_map,
            hover_name="grid_id",
            hover_data=hover_data,
            size_max=point_size,
            zoom=zoom,
            center={"lat": center_lat, "lon": center_lon},
            map_style="carto-positron",
            title=f"<b>{title}</b>",
            height=680
        )
        ScatterTrace = go.Scattermap
    else:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="cluster_name",
            color_discrete_map=color_map,
            hover_name="grid_id",
            hover_data=hover_data,
            size_max=point_size,
            zoom=zoom,
            center={"lat": center_lat, "lon": center_lon},
            mapbox_style="carto-positron",
            title=f"<b>{title}</b>",
            height=680
        )
        ScatterTrace = getattr(go, "Scattermapbox", go.Scattermap)

    # Enhance marker size & opacity
    fig.update_traces(
        marker=dict(size=point_size, opacity=0.88),
        selector=dict(mode="markers")
    )

    # Add hotspot halo ring for extreme cluster
    hotspot_df = df[df["is_hotspot"]]
    if len(hotspot_df) > 0:
        fig.add_trace(
            ScatterTrace(
                lat=hotspot_df["lat"],
                lon=hotspot_df["lon"],
                mode="markers",
                marker=dict(
                    size=point_size + 4,
                    color="rgba(127, 0, 0, 0.4)",
                    opacity=0.7
                ),
                name="Hotspot Boundary Aura",
                hoverinfo="skip",
                showlegend=True
            )
        )

    fig.update_layout(
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        legend=dict(
            title=dict(text="<b>Heat Severity Tier</b>"),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.88)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        )
    )

    return fig


def plot_elbow_and_silhouette(
    diagnostics_df: pd.DataFrame,
    selected_k: int = 4
) -> go.Figure:
    """
    Renders side-by-side interactive chart: Inertia (Elbow) vs Silhouette Score across k=2..10.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Elbow Curve: Inertia (WCSS) vs k</b>", "<b>Silhouette Score vs k (Clustering Quality)</b>")
    )

    # Inertia curve
    fig.add_trace(
        go.Scatter(
            x=diagnostics_df["k"],
            y=diagnostics_df["inertia"],
            mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=9, color="#1f77b4")
        ),
        row=1, col=1
    )

    # Silhouette curve
    fig.add_trace(
        go.Scatter(
            x=diagnostics_df["k"],
            y=diagnostics_df["silhouette_score"],
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=9, color="#2ca02c")
        ),
        row=1, col=2
    )

    # Highlight selected k
    sel_row = diagnostics_df[diagnostics_df["k"] == selected_k]
    if len(sel_row) > 0:
        sel_inertia = sel_row["inertia"].values[0]
        sel_sil = sel_row["silhouette_score"].values[0]

        fig.add_trace(
            go.Scatter(
                x=[selected_k],
                y=[sel_inertia],
                mode="markers",
                marker=dict(size=15, color="#d62728", symbol="star"),
                name=f"Selected k={selected_k}",
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[selected_k],
                y=[sel_sil],
                mode="markers",
                marker=dict(size=15, color="#d62728", symbol="star"),
                name=f"Selected k={selected_k}",
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="Inertia (Within-Cluster Sum of Squares)", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig.update_layout(
        height=380,
        margin={"r": 20, "t": 40, "l": 40, "b": 40},
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )

    return fig


def plot_clustering_diagnostics(
    diagnostics_df: pd.DataFrame,
    selected_k: int = 4
) -> go.Figure:
    """
    Renders 4-panel interactive diagnostic chart across k=2..8:
      1. Inertia / WCSS (Elbow curve)
      2. Silhouette Score (clustering quality)
      3. Davies-Bouldin Index (cluster compactness / separation, lower is better)
      4. Calinski-Harabasz Index (variance ratio, higher is better)
    Highlights selected k with prominent star markers.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>1. Elbow Curve: Inertia (WCSS) vs k</b>",
            "<b>2. Silhouette Score vs k (Higher is Better)</b>",
            "<b>3. Davies-Bouldin Index vs k (Lower is Better)</b>",
            "<b>4. Calinski-Harabasz Index vs k (Higher is Better)</b>"
        )
    )

    k_vals = diagnostics_df["k"]

    # 1. Inertia
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=diagnostics_df["inertia"], mode="lines+markers",
            name="Inertia (WCSS)", line=dict(color="#1f77b4", width=3),
            marker=dict(size=8, color="#1f77b4")
        ),
        row=1, col=1
    )

    # 2. Silhouette
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=diagnostics_df["silhouette_score"], mode="lines+markers",
            name="Silhouette Score", line=dict(color="#2ca02c", width=3),
            marker=dict(size=8, color="#2ca02c")
        ),
        row=1, col=2
    )

    # 3. Davies-Bouldin
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=diagnostics_df["davies_bouldin"], mode="lines+markers",
            name="Davies-Bouldin Index", line=dict(color="#ff7f0e", width=3),
            marker=dict(size=8, color="#ff7f0e")
        ),
        row=2, col=1
    )

    # 4. Calinski-Harabasz
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=diagnostics_df["calinski_harabasz"], mode="lines+markers",
            name="Calinski-Harabasz Index", line=dict(color="#9467bd", width=3),
            marker=dict(size=8, color="#9467bd")
        ),
        row=2, col=2
    )

    # Highlight selected k
    sel_row = diagnostics_df[diagnostics_df["k"] == selected_k]
    if len(sel_row) > 0:
        val_inertia = sel_row["inertia"].values[0]
        val_sil = sel_row["silhouette_score"].values[0]
        val_db = sel_row["davies_bouldin"].values[0]
        val_ch = sel_row["calinski_harabasz"].values[0]

        for r, c, val in [
            (1, 1, val_inertia),
            (1, 2, val_sil),
            (2, 1, val_db),
            (2, 2, val_ch)
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[selected_k], y=[val], mode="markers",
                    marker=dict(size=14, color="#d62728", symbol="star"),
                    name=f"Selected k={selected_k}",
                    showlegend=(r == 1 and c == 1)
                ),
                row=r, col=c
            )

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Number of Clusters (k)", row=r, col=c, dtick=1)

    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_yaxes(title_text="Davies-Bouldin Index", row=2, col=1)
    fig.update_yaxes(title_text="Calinski-Harabasz Index", row=2, col=2)

    fig.update_layout(
        height=660,
        margin={"r": 20, "t": 50, "l": 40, "b": 40},
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    return fig


def plot_cluster_radar(
    profiles_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> go.Figure:
    """
    Renders radar / polar chart comparing the 4 normalized core heat feature footprints across clusters:
    Mean Tmax, Max Tmax, Days >= 40°C, and Heatwave Days.
    """
    feat_cols = features or ["tmax_mean", "tmax_max", "hot_days_ge_40", "heatwave_days"]
    friendly_names = {
        "tmax_mean": "Mean Tmax (°C)",
        "tmax_max": "Max Tmax (°C)",
        "hot_days_ge_40": "Days >= 40°C",
        "heatwave_days": "Heatwave Days"
    }

    norm_df = profiles_df.copy()
    categories = [friendly_names.get(c, c) for c in feat_cols]

    fig = go.Figure()

    for _, row in norm_df.iterrows():
        cname = row["cluster_name"]
        color = SEVERITY_COLORS.get(cname, "#555555")

        values = []
        for col in feat_cols:
            max_val = profiles_df[col].max()
            val = (row[col] / max_val * 100.0) if max_val > 0 else 0
            values.append(val)
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=cname,
                line=dict(color=color, width=2.5),
                opacity=0.6
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 105], ticksuffix="%")
        ),
        title="<b>Cluster Thermal Footprint Comparison (Normalized Radar Profile)</b>",
        height=450,
        margin={"r": 30, "t": 60, "l": 30, "b": 30},
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
    )

    return fig


def plot_feature_distributions(
    labeled_df: pd.DataFrame,
    feature: str = "hot_days_ge_40",
    feature_label: str = "Hot Days (Tmax ≥ 40°C)"
) -> go.Figure:
    """
    Renders box plots comparing the distribution of a specific feature across cluster tiers.
    """
    color_map = {name: SEVERITY_COLORS.get(name, "#555555") for name in labeled_df["cluster_name"].unique()}

    # Order categories by cluster_id
    cat_order = labeled_df.sort_values("cluster_id")["cluster_name"].unique().tolist()

    fig = px.box(
        labeled_df,
        x="cluster_name",
        y=feature,
        color="cluster_name",
        color_discrete_map=color_map,
        category_orders={"cluster_name": cat_order},
        points="all",
        title=f"<b>Distribution of {feature_label} by Cluster Severity Tier</b>",
        labels={"cluster_name": "Cluster Severity", feature: feature_label},
        height=380
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        margin={"r": 20, "t": 45, "l": 40, "b": 40}
    )

    return fig


def plot_region_breakdown(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Renders a normalized 100% stacked horizontal bar chart showing the composition
    of each meteorological region across cluster severity tiers.
    """
    ct = pd.crosstab(labeled_df["region"], labeled_df["cluster_name"], normalize="index") * 100
    ct = ct.reset_index()

    melted = ct.melt(id_vars="region", var_name="cluster_name", value_name="percentage")

    color_map = {name: SEVERITY_COLORS.get(name, "#555555") for name in labeled_df["cluster_name"].unique()}

    fig = px.bar(
        melted,
        y="region",
        x="percentage",
        color="cluster_name",
        color_discrete_map=color_map,
        orientation="h",
        title="<b>Regional Distribution across Heat Severity Tiers (% of Grid Cells)</b>",
        labels={"percentage": "% Grid Cells in Region", "region": "IMD Region", "cluster_name": "Severity Tier"},
        height=380
    )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        margin={"r": 20, "t": 45, "l": 40, "b": 40},
        xaxis=dict(ticksuffix="%")
    )

    return fig


if __name__ == "__main__":
    from src.cluster import fit_kmeans_and_label, evaluate_k_range
    feat_file = Path("data/processed/features_heatwave_season.parquet")
    df_feat = pd.read_parquet(feat_file)
    labeled, profs, meta = fit_kmeans_and_label(df_feat, k=4)
    fig_map = plot_hotspot_map(labeled)
    print("plot_hotspot_map built successfully:", type(fig_map))
    fig_radar = plot_cluster_radar(profs)
    print("plot_cluster_radar built successfully:", type(fig_radar))
    diag_df = evaluate_k_range(df_feat, k_min=2, k_max=8)
    fig_diag = plot_clustering_diagnostics(diag_df, selected_k=4)
    print("plot_clustering_diagnostics built successfully:", type(fig_diag))
