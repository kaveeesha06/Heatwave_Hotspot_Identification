# Heatwave Hotspot Identification Using K-Means Clustering

Mini project aligned with AI Use Case **KJS-CES-01: Climate Intelligence for Heatwave Monitoring, Prediction, and Early Warning** (collaborating organization: India Meteorological Department, Mumbai-Pune).

This project addresses **Phase II (AI-driven heatwave analytics: hotspot identification and severity classification)** of the use case and demonstrates a lightweight version of the **Phase V decision-support dashboard**.

---

## 1. Problem Statement

Heatwave occurrence and severity vary widely across regions of India. Looking at raw temperature maps makes it hard to see which areas behave similarly and which are persistent heat hotspots. This project groups grid cells across India into clusters with similar heat behavior using **K-Means**, then identifies and ranks the clusters so that the hottest ones can be flagged as **heatwave hotspots**.

## 2. Objectives

1. Load and clean IMD gridded maximum temperature data.
2. Engineer per-grid-cell features that describe heat behavior (not just raw temperature).
3. Apply K-Means clustering and choose the number of clusters (k) using the elbow method and silhouette score.
4. Label clusters by severity (e.g., Low, Moderate, High, Extreme).
5. Present results on an interactive map and dashboard.
6. Validate the hotspots against known heatwave-prone regions of India.

## 3. Dataset

- **Source:** IMD Pune, gridded daily maximum temperature (GRD files)
  https://imdpune.gov.in/lrfindex.php
- **Attributes:** Timestamp, Latitude, Longitude, Maximum Temperature
- **Missing values:** IMD uses a fill value (about 99.9) for no data; these are converted to `NaN`.
- Check the IMD documentation for the exact grid resolution of the product you download.

> Raw data is not committed to the repository. Download it from the IMD link above into `data/raw/`.

## 4. Final Tech Stack

The table below lists the final technologies and libraries utilized in this project:

| Layer | Tool / Library | Role & Specific Purpose |
|---|---|---|
| **Programming Language** | Python (3.10+) | Core programming language for data pipelines, modeling, and dashboard |
| **Data Ingestion** | `imdlib` | Communicates with IMD Pune servers, downloads `.GRD` binaries, and parses binary grids |
| **Array Computing** | `numpy` | Multi-dimensional grid arrays, binary reshaping, and numerical operations |
| **Gridded Data Handling** | `xarray` | Multidimensional climate dataset representation across `(time, lat, lon)` |
| **Data Manipulation** | `pandas` | Tabular data transformations, temporal filtering, and grid-level feature aggregations |
| **Machine Learning** | `scikit-learn` | `StandardScaler` (feature scaling), `KMeans` (clustering), and clustering metrics (`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`) |
| **Feature Storage** | `pyarrow` (Apache Parquet) | High-performance, columnar disk caching for engineered feature tables in `data/processed/` |
| **Interactive Visualization** | `plotly` (`plotly.express` & `plotly.graph_objects`) | Geospatial scatter maps (`scatter_map`), normalized radar profiles, and dual-axis diagnostic plots |
| **Web Dashboard** | `streamlit` | Modern, single-page reactive decision-support dashboard (`app.py`) |
| **Exploration & Reporting** | Jupyter Notebook (`notebooks/`) | Step-by-step EDA, k-selection experiments, and meteorological validation report |

## 5. How It Works

```
IMD GRD files -> imdlib -> cleaning -> feature engineering
-> scaling -> K-Means (k via elbow/silhouette) -> cluster labeling
-> map visualization -> Streamlit dashboard
```

### Step 1: Data acquisition
Download Tmax GRD files from IMD and load them with `imdlib`, giving daily Tmax over India.

### Step 2: Preprocessing
- Replace fill values with `NaN`.
- Restrict to the heatwave season (March to June) and a chosen range of years.
- Flatten to one row per grid cell.

### Step 3: Feature engineering
K-Means on raw Tmax gives uninteresting results, so features are built per grid cell:

- Mean seasonal Tmax
- Maximum Tmax recorded
- Number of days with Tmax >= 40 C
- Number of heatwave days (IMD criteria: Tmax >= 40 C in plains and at least 4.5 C above normal, or Tmax >= 45 C)
- Longest consecutive hot-day streak
- Standard deviation of Tmax

Latitude and longitude are deliberately **excluded** so that clusters reflect heat behavior rather than geography. Including them would force spatially contiguous clusters.

### Step 4: Scaling
`StandardScaler` is applied because K-Means is distance-based and sensitive to feature scale.

### Step 5: Choosing k
Run K-Means for k = 2 to 10 and compare the elbow curve (inertia) and silhouette score. Typically k between 3 and 5 works well; the final choice is justified in the report.

### Step 6: Clustering and labeling
Fit K-Means, rank clusters by mean hot-day count or mean Tmax, and label them (Low, Moderate, High, Extreme). The top cluster is reported as the **heatwave hotspot**. This provides a simple severity classification in line with the use case.

### Step 7: Dashboard
The Streamlit app shows:
- Map of India with grid cells colored by cluster
- Sidebar filters: year, month/season, number of clusters (k)
- Cluster profile table (average Tmax, hot days per cluster)
- Elbow and silhouette plots
- Optional: region-wise view using the seven IMD-defined regions

### Step 8: Validation
- Compare the extreme cluster with known heatwave-prone zones (central India, Rajasthan, Vidarbha, Odisha coast, Andhra Pradesh).
- Check cluster stability across different years.
- Report silhouette score and Davies-Bouldin index.

## 6. Project Structure

```
heatwave-kmeans/
├── data/
│   ├── raw/            # IMD GRD files (not committed)
│   └── processed/      # Feature tables (parquet)
├── notebooks/          # EDA and k-selection experiments
├── src/
│   ├── load_data.py    # Read GRD files, clean, filter
│   ├── features.py     # Per-grid-cell feature engineering
│   ├── cluster.py      # Scaling, K-Means, evaluation, labeling
│   └── plots.py        # Map and diagnostic plots
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 7. Execution Sequence & Usage Guide

Follow the commands below in sequential order to run the entire pipeline from data ingestion to the interactive dashboard.

### Quick Start (Run in 3 Steps)

If your environment is already set up:
```bash
# Step 1: Generate features
python src/features.py

# Step 2: Run clustering & validation
python src/cluster.py

# Step 3: Launch dashboard
python -m streamlit run app.py
```

---

### Detailed Step-by-Step Sequence

#### Step 1: Environment Setup
Open your terminal (PowerShell or Bash) in the project root directory:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

#### Step 2: Download & Ingest IMD Gridded Temperature Data
Fetches official IMD 1.0° daily maximum temperature binary (`.GRD`) files, masks sentinel ocean values (`99.9`), and cleans the grid to mainland India (355 land grid cells):

```bash
python src/load_data.py
```
> **Output:** Raw files stored in `data/raw/tmax/` (`2022.GRD`, `2023.GRD`, `2024.GRD`). Includes an offline synthetic generator fallback if IMD servers are unreachable.

#### Step 3: Engineer Per-Grid-Cell Thermal Features
Computes 6 seasonal heat behavior metrics (Mean $T_{max}$, Peak $T_{max}$, Days $\ge 40^\circ\text{C}$, Heatwave days, Max hot streak, Temp volatility) over the March–June heatwave season:

```bash
python src/features.py
```
> **Output:** Caches clean feature tables in `data/processed/` (`features_heatwave_season.parquet` and individual year parquets).

#### Step 4: Run K-Means Clustering & Scientific Diagnostics
Standardizes features with `StandardScaler`, sweeps $k \in [2, 8]$ computing **Inertia (Elbow)** and **Silhouette scores**, fits $k=4$, and validates against known heatwave zones:

```bash
python src/cluster.py
```
> **Output:** Prints Elbow/Silhouette metrics table, cluster profile summaries, and cross-tabulation with IMD meteorological regions (Vidarbha, Rajasthan, Himalayas).

#### Step 5: (Optional) Verify Interactive Plot Functions
Verifies Plotly map rendering, radar footprints, and diagnostic charts:

```bash
python src/plots.py
```

#### Step 6: Launch the Interactive Streamlit Dashboard
Starts the single-page decision-support web application in your browser:

```bash
python -m streamlit run app.py
```
> **URL:** Open your browser at **`http://localhost:8501`** to interact with the map, filters, and early warning advisories.

#### Step 7: (Optional) Explore the Jupyter Notebook
For interactive step-by-step experimentation and report generation:

```bash
jupyter notebook notebooks/01_eda_and_clustering.ipynb
```
*(or open `notebooks/01_eda_and_clustering.ipynb` directly in VS Code).*

## 8. Limitations

- K-Means assumes roughly spherical clusters and requires k to be fixed in advance.
- It does not model spatial adjacency directly (DBSCAN or Gaussian Mixture models are natural comparisons).
- It identifies areas that are **consistently hot**, not individual heatwave events. For event-level hotspots, cluster on a specific heatwave period (for example, April to May 2024).
- Hotspots are derived from historical gridded data and are not a forecast.

## 9. Future Work

- Compare K-Means with DBSCAN and Gaussian Mixture Models.
- Add a temporal view showing how hotspots shift from March to June.
- Generate stakeholder-specific advisories per hotspot using an LLM (Phase V of the use case), with a human-review step before dissemination, in line with the Human-in-the-Loop governance requirement.
- Integrate localized IoT Automated Weather Station (AWS) observations to validate hotspots (Phases III and IV).

## 10. Responsible AI Note

This project is an academic prototype. Cluster labels and any advisories are **not official warnings**. Official heatwave forecasts and advisories should be taken from the India Meteorological Department. Uncertainty (silhouette score, stability across years) should always be reported alongside results.

## 11. Team

| Name | Roll No. | Role |
|---|---|---|
| | | |
| | | |
| | | |

**Guide:** Dr. Radhika Kotecha, Professor and Head, Department of Information Technology, K J Somaiya Institute of Technology
*(update if your mini project guide is different)*

## 12. Acknowledgements

- India Meteorological Department (IMD), Pune for the gridded temperature data
- K J Somaiya Institute of Technology, Framework for AI Use Case Integration in Curriculum Delivery (Use Case KJS-CES-01)
