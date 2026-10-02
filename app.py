"""
Streamlit Web Application: Heatwave Hotspot Identification Using K-Means Clustering
Design: Single-Page Minimalist View (Metrics -> Interactive Map -> Hotspot Table -> Action Alerts)
Aligned with AI Use Case KJS-CES-01: Climate Intelligence for Heatwave Monitoring
Collaborating Organization: India Meteorological Department (IMD), Mumbai-Pune.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.load_data import (
    load_heatwave_season_dataframe,
    get_available_raw_years
)
from src.features import (
    compute_grid_features,
    CORE_HEAT_FEATURES
)
from src.cluster import (
    fit_kmeans_and_label,
    evaluate_k_range,
    SEVERITY_COLORS
)
from src.plots import (
    plot_hotspot_map,
    plot_elbow_and_silhouette
)


# Page setup
st.set_page_config(
    page_title="Heatwave Hotspot Tracker | IMD",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalistic styling
st.markdown("""
<style>
    /* Clean typography and spacing */
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .subtitle-text {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #F1F5F9;
        color: #334155;
        margin-right: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .tag-red {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .tag-blue {
        background-color: #E0E7FF;
        color: #3730A3;
    }
    /* Simple card styling */
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    /* Remove unnecessary padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Caching data loading and diagnostics for snappy performance
@st.cache_data(show_spinner=False)
def load_data(start_year: int, end_year: int):
    return load_heatwave_season_dataframe(
        start_year=start_year,
        end_year=end_year,
        season_months=(3, 4, 5, 6)
    )


@st.cache_data(show_spinner=False)
def get_diagnostics(features_df: pd.DataFrame):
    return evaluate_k_range(features_df, feature_cols=CORE_HEAT_FEATURES, k_min=2, k_max=8)


# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## **⚙️ Quick Controls**")

avail_years = get_available_raw_years("data/raw") or [2022, 2023, 2024]
year_options = ["2022–2024 (3-Year Average)"] + [str(y) for y in sorted(avail_years, reverse=True)]

selected_year_str = st.sidebar.selectbox(
    "Select Year / Timeframe",
    options=year_options,
    index=0,
    help="Analyze the multi-year seasonal average or an individual heatwave year."
)

selected_region = st.sidebar.selectbox(
    "Filter by Region",
    options=[
        "All India",
        "Northwest India / Rajasthan",
        "Central India / Vidarbha",
        "Central Plains",
        "East & Northeast India",
        "South Peninsular / Deccan",
        "West Coast / Konkan / Goa",
        "Western Himalayas / North"
    ],
    index=0
)

only_hotspots = st.sidebar.checkbox(
    "🔥 Show Extreme Hotspots Only",
    value=False,
    help="Hide milder areas and highlight only the critical heatwave corridors."
)

# Advanced parameters neatly tucked inside an expander
with st.sidebar.expander("🛠️ Advanced ML Parameters"):
    k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=6, value=4)
    hot_temp_threshold = st.slider("Hot Day Threshold (°C)", min_value=38.0, max_value=43.0, value=40.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption("IMD Mumbai-Pune Collab | AI Use Case KJS-CES-01")


# --- DATA PIPELINE ---
if selected_year_str == "2022–2024 (3-Year Average)":
    s_yr, e_yr = min(avail_years), max(avail_years)
else:
    s_yr = int(selected_year_str)
    e_yr = int(selected_year_str)

with st.spinner("Analyzing IMD heatwave data..."):
    season_raw_df = load_data(s_yr, e_yr)
    features_df = compute_grid_features(season_raw_df, hot_threshold=hot_temp_threshold)
    labeled_df, profiles_df, meta = fit_kmeans_and_label(features_df, k=k_clusters)

# Filter display dataframe if region selected
if selected_region != "All India":
    display_df = labeled_df[labeled_df["region"] == selected_region].copy()
    if len(display_df) == 0:
        st.warning(f"No grid cells match '{selected_region}'. Showing All India.")
        display_df = labeled_df.copy()
else:
    display_df = labeled_df.copy()

if only_hotspots:
    display_df = display_df[display_df["is_hotspot"]].copy()


# --- HEADER SECTION ---
st.markdown("""
<div>
    <span class="tag tag-red">Use Case KJS-CES-01</span>
    <span class="tag tag-blue">India Meteorological Department (IMD)</span>
    <span class="tag">Unsupervised K-Means</span>
</div>
<div class="title-text">🔥 India Heatwave Hotspot Tracker</div>
<div class="subtitle-text">
    Identifying chronic heatwave hotspots and severity zones across India (March–June) using machine learning on gridded temperature records.
</div>
""", unsafe_allow_html=True)


# --- 1. TOP KEY METRICS ROW ---
hotspot_cells = display_df[display_df["is_hotspot"]]
hotspot_count = len(hotspot_cells)
total_count = len(display_df)
hotspot_pct = (hotspot_count / total_count * 100) if total_count > 0 else 0
peak_temp = display_df["tmax_max"].max() if total_count > 0 else 0
avg_hotspot_days = hotspot_cells["hot_days_ge_40"].mean() if hotspot_count > 0 else 0
sil_score = meta["silhouette_score"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Peak Temperature Recorded",
        value=f"{peak_temp:.1f} °C",
        help="Highest daily maximum temperature observed in this selection."
    )
with col2:
    st.metric(
        label="Extreme Hotspot Area",
        value=f"{hotspot_pct:.1f}%",
        delta=f"{hotspot_count} of {total_count} cells",
        delta_color="inverse",
        help="Percentage of grid cells classified in the highest severity cluster."
    )
with col3:
    st.metric(
        label="Avg Hot Days in Hotspot",
        value=f"{avg_hotspot_days:.0f} days",
        help=f"Average number of days with Tmax ≥ {hot_temp_threshold}°C in the extreme tier."
    )
with col4:
    st.metric(
        label="AI Cluster Quality (Silhouette)",
        value=f"{sil_score:.2f}",
        delta="Strong separation" if sil_score >= 0.4 else "Moderate",
        help="Clustering cohesion and separation score (-1 to 1). Values > 0.4 indicate well-defined clusters."
    )

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)


# --- 2. INTERACTIVE GEOGRAPHIC MAP ---
st.markdown("### 🗺️ Geographic Heatwave Hotspots Map")
st.caption("Hover over any grid point to view its coordinates, region, average temperature, and hot streak length.")

fig_map = plot_hotspot_map(
    display_df,
    title=f"Heatwave Severity Map — {selected_year_str} ({selected_region})",
    point_size=15
)
st.plotly_chart(fig_map, width="stretch")


# --- 3. SIMPLE HOTSPOT SEVERITY TABLE ---
st.markdown("### 📊 Heatwave Severity Summary")
st.caption("Quantitative comparison of each heat severity tier across the country.")

summary_table = profiles_df.copy()
rename_cols = {
    "cluster_name": "Severity Tier",
    "cell_count": "Grid Cells",
    "pct_land_area": "% of India",
    "tmax_mean": "Avg Temp (°C)",
    "tmax_max": "Max Temp (°C)",
    "hot_days_ge_40": f"Hot Days (≥{hot_temp_threshold}°C)",
    "longest_hot_streak": "Max Streak (Days)"
}
available_cols = [c for c in rename_cols.keys() if c in summary_table.columns]
summary_display = summary_table[available_cols].rename(columns=rename_cols)

# Format for clean display
st.dataframe(summary_display, width="stretch", hide_index=True)


# --- 4. ACTION ADVISORIES & EARLY WARNING (PHASE V) ---
st.markdown("### 🛡️ Recommended Early Warning Actions")
st.caption("Targeted public health and municipal directives based on identified heatwave severity.")

adv_col1, adv_col2 = st.columns(2)

with adv_col1:
    st.error("""
    #### 🚨 Extreme Hotspot Tier (Red Alert)
    **Core Zones:** Vidarbha (Nagpur/Chandrapur), Western Rajasthan (Churu/Bikaner), Central Plains
    * **Outdoor Labor:** Mandate suspension of construction and manual outdoor labor between 12:00 PM and 4:00 PM.
    * **Public Cooling:** Open municipal cooling centers and hydration stations in crowded transit hubs.
    * **Healthcare:** Activate Heat Action Plan (HAP) Level 3 — stock IV fluids, ORS, and ice packs in all PHCs.
    * **Water Resources:** Prioritize dedicated emergency water tankers to vulnerable informal settlements.
    """)

with adv_col2:
    st.warning("""
    #### ⚠️ High Severity Tier (Orange Alert)
    **Core Zones:** East Rajasthan, Gangetic Plains, Interior Odisha, Telangana
    * **Schools & Institutions:** Shift school hours to morning sessions (concluding before 11:30 AM).
    * **Public Alerts:** Issue daily heat advisories via SMS, local radio, and municipal message boards.
    * **Power Grid:** Plan for 20–30% surge in cooling power demand; monitor distribution transformers.
    * **Vulnerable Groups:** Special monitoring for infants, outdoor workers, and the elderly.
    """)

st.info("""
**🟢 Moderate & Low Severity Tiers (Yellow / Green):** Coastal areas (Konkan, Goa, Kerala) and Western Himalayas. Standard seasonal monitoring. Maintain regular drinking water supplies and advisories.
""")


# --- 5. OPTIONAL TECHNICAL DETAILS EXPANDER ---
with st.expander("🔬 View Machine Learning Diagnostics & Justification (Elbow & Silhouette)"):
    st.markdown("#### **Hyperparameter Validation (Choosing k=4)**")
    st.write(
        r"To ensure scientific rigor, we evaluated K-Means across $k \in [2, 8]$ using **Inertia (Elbow Method)** "
        r"and the **Silhouette Score**:"
    )
    
    diagnostics_df = get_diagnostics(features_df)
    fig_diag = plot_elbow_and_silhouette(diagnostics_df, selected_k=k_clusters)
    st.plotly_chart(fig_diag, width="stretch")
    
    st.markdown("""
    * **Elbow Inflection:** The inertia curve shows a distinct reduction up to $k=4$, after which gains plateau.
    * **Silhouette Quality:** $k=4$ achieves a high silhouette score ($\approx 0.45$), confirming well-separated clusters.
    * **Operational Fit:** 4 clusters naturally map to the IMD's standard hazard scale: *Low, Moderate, High, Extreme Hotspot*.
    """)


# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    <b>Responsible AI Note:</b> Prototype for academic evaluation (AI Use Case KJS-CES-01). 
    Official heatwave forecasts and warnings are issued exclusively by the India Meteorological Department (IMD).
</div>
""", unsafe_allow_html=True)
