<p align="center">
  <img src="sonality-lockup.png" alt="Sonality" width="380">
</p>

<h1 align="center">Sonality</h1>
<p align="center">
  <strong>What's your sound? Unsupervised Music Taste Profiler & Full-Stack ML Engine</strong>
</p>

<p align="center">
  <a href="https://github.com/RyWidjaya/sonality"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg" alt="Scikit-Learn"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-Backend%20API-teal.svg" alt="FastAPI"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Containerized-2496ED.svg" alt="Docker"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

## ✨ What is Sonality?

**Sonality** is an applied machine learning web application that analyzes a listener's music taste independent of canned, human-assigned genre tags (e.g., "Rock" vs. "Pop"). 

By evaluating continuous acoustic audio features across **127,900+ curated tracks**, Sonality groups music into **7 Universal Sound Personas** using unsupervised learning ($k$-Means + Yeo-Johnson PowerTransform + PCA), benchmarks listener habits against the global listening population, and delivers within-cluster recommendations with instant audio previews.

### Key Interactive Features
* 🗺️ **Interactive 2D Sound Map**: Projects your selected songs in real-time onto a 2D PCA landscape alongside 127,900 background tracks.
* 🧭 **Acoustic Radar & Trait Percentiles**: Compares your acoustic traits (*tempo, danceability, valence, and intensity*) against the entire catalog using empirical percentiles.
* ⚖️ **Shannon Diversity Metric**: Quantifies how focused vs. eclectic your music taste is using normalized Shannon information entropy:
  $$H = -\frac{\sum p_i \ln(p_i)}{\ln(7)}$$
* 🎧 **Intelligent Recommendations & Audio Previews**: Discovers nearest-neighbor tracks within your primary persona cluster, complete with integrated 30-second audio previews powered by Apple Music.
* 📱 **Editorial 4:5 Persona Share Card**: High-resolution, container-query responsive shareable graphic rendered entirely client-side using `html-to-image` for one-click downloading and social sharing.

---

## 📌 Why This Project?

Most music recommendation projects build supervised classifiers to predict existing `track_genre` labels. That is simply classification wearing a clustering costume.

Sonality takes a **strictly unsupervised approach**: rather than relying on noisy, subjective marketing labels, it lets **$k$-Means discover its own sound categories** directly from continuous acoustic features. The result is a system that evaluates a real listener's taste profile based on sound physics, tempo, and timbre rather than artist branding.

```
                  ┌─────────────────────────────────────────┐
                  │    Spotify Audio Data (127,900 Tracks)  │
                  └────────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │ Yeo-Johnson Transform │             │    1D PCA Intensity   │
        │   (PowerTransformer)  │             │(Energy/Loud/Acoustic) │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     7-D Standardized Space    │
                       │   (Dance, Speech, Valence...) │
                       └───────────────┬───────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
          ┌───────────────────────┐         ┌───────────────────────┐
          │  k-Means Clustering   │         │    2D PCA Projector   │
          │    (k = 7 Personas)   │         │  (Interactive Sound)  │
          └───────────┬───────────┘         └───────────┬───────────┘
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │         Serialized Model Brain          │
                  │   (Joblib weights & Persona Metadata)   │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │      FastAPI High-Speed REST API        │
                  │  (/api/taste-profile & /api/search)     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        Sonality Web Dashboard           │
                  │  (Sound Map, Radar, Diversity, 4:5 Card)│
                  └─────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline & Engineering Decisions

### 1. Data Cleaning & Modern Catalog Augmentation
* **Deduplication of Baseline Data**: The raw 114k Spotify dataset contained ~24,000 duplicate tracks due to cross-genre tagging (e.g., the same song labeled as both "pop" and "dance"). Retaining duplicates would have silently pulled cluster centroids toward over-represented commercial tracks. Cleaned strictly down to **89,583 unique baseline songs**.
* **Incremental Ingestion & Catalog Modernization**: Curated recent high-popularity releases from warehouse audio feature datasets to modernize representation with 0 nulls across **127,900 unique tracks**.

### 2. Distribution Correction (Yeo-Johnson)
* Features like `speechiness`, `instrumentalness`, and `liveness` exhibit heavy positive skew and zero-inflation.
* Replaced standard linear scaling with the **Yeo-Johnson PowerTransformer** to stabilize variance and normalize feature distributions without clipping or data loss.

### 3. Collinearity Resolution (1D PCA Intensity Triad)
* Exploratory data analysis revealed high pairwise collinearity between `energy`, `loudness`, and `acousticness` ($|r| > 0.70$).
* Clustering on these separately would cause $k$-Means distance metrics to over-index on loudness/volume at the expense of rhythmic and structural features.
* Applied **Principal Component Analysis (PCA)** to collapse this triad into a single composite feature: **`intensity`** (capturing ~78% of shared variance).

### 4. Model Selection: Why $k = 7$?
* Evaluated clusters across $k \in [2, 10]$ using **Elbow inertia** and **Silhouette analysis**.
* While $k=5$ yielded a marginally higher silhouette score ($0.173$ vs $0.166$), inspecting cluster composition revealed severe semantic over-compression: ambient sleep tracks were lumped with acoustic ballads, and techno was grouped with death metal.
* **$k = 7$ was chosen** because it cleanly separated distinct acoustic genres while retaining statistical coherence and high domain interpretability.

---

## 🎧 The 7 Universal Sound Personas

| Cluster | Sound Persona | Distinguishing Acoustic Signature | Common Genres / Representative Tracks |
| :---: | :--- | :--- | :--- |
| **0** | **Heavy & Live** | Ultra-high intensity (+1.55), High liveness (+0.70), High tempo (+0.57) | Metalcore, Heavy Metal, Hard Rock, Punk |
| **1** | **Driven Alt & Punk** | High tempo (+0.80), Low intensity (-0.98), Low speechiness (-0.66) | Alternative, Indie Rock, Dark Wave, Post-Punk |
| **2** | **Synth & Club** | Ultra-high instrumentalness (+1.87), High intensity (+0.86) | Techno, House, EDM, Breakbeat, Trance |
| **3** | **Vocal & Rhythmic** | High speechiness (+1.52), High danceability (+0.83) | Hip-Hop, Rap, Dancehall, Trap |
| **4** | **Bright Pop & Groove** | High valence (+0.83), High danceability (+0.69), High intensity (+0.69) | Dance-Pop, Disco, Salsa, Funk, Upbeat Pop |
| **5** | **Unplugged & Intimate** | Low intensity (-1.31), Low tempo (-1.02), Low speechiness (-0.65) | Acoustic, Folk, Singer-Songwriter, Ballads |
| **6** | **Ambient & Drone** | Ultra-low intensity (-2.70), High instrumentalness (+1.90), Low valence (-1.09) | Ambient, Classical, Sleep, New-Age, Drone |

---

## 📂 Repository Structure

```
├── backend/                        # FastAPI backend & ML inference pipeline
│   ├── main.py                     # API routes, static file serving & Apple preview streaming
│   ├── engine.py                   # PowerTransformer inference, PCA projections, recommendation math
│   └── schemas.py                  # Pydantic request/response validation schemas
├── data/
│   └── spotify_tracks_modeled.csv  # Production catalog (127.9k modeled tracks with personas & 2D PCA)
├── frontend/                       # Vanilla editorial frontend (zero heavy runtime frameworks)
│   ├── index.html                  # Semantic layout, modal structures & accessible controls
│   ├── styles.css                  # Custom dark editorial design & 4:5 CSS container queries
│   ├── app.js                      # Canvas 2D sound map, radar chart, Apple preview player & state
│   ├── html-to-image.js            # Vendored client-side 4:5 card PNG export library
│   ├── sonality-mark.svg           # Brand vector glyph
│   └── favicon.svg                 # Browser favicon
├── models/                         # Serialized ML model weights & metadata
│   ├── kmeans_model.joblib         # Fitted 7-cluster K-Means model
│   ├── power_transformer.joblib    # Fitted Yeo-Johnson transformer
│   ├── pca_intensity.joblib        # 1D PCA intensity triad model (energy/loudness/acousticness)
│   ├── pca_2d.joblib               # 2D PCA sound-map projector
│   └── persona_metadata.json       # Persona labels, colors, descriptions, and feature schemas
├── notebooks/                      # End-to-end data science research & exploration
│   ├── 01_cleaning.ipynb           # Baseline data audit, deduplication & null handling
│   ├── 02_eda.ipynb                # Exploratory analysis, distributions & multicollinearity diagnostics
│   ├── 03_data_augmentation.ipynb  # Catalog modernization & feature alignment
│   └── 04_ml_pipeline.ipynb        # Production Yeo-Johnson, 1D PCA, KMeans & model serialization
├── Dockerfile                      # Production container configuration (Google Cloud Run / Render)
├── .dockerignore                   # Container build exclusions
├── .gitignore                      # Git tracking exclusions (pycache, intermediate CSVs, logs)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Quickstart (Local Development)

> **Deploying as a live website?** You do not need to install Python locally. The web app is fully containerized via `Dockerfile` and deploys in one click to **Google Cloud Run** or **Render**.

If you want to run the project locally on your machine:

### 1. Clone & Setup
```bash
git clone https://github.com/RyWidjaya/sonality.git
cd sonality
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Application
```bash
uvicorn backend.main:app --reload
```
Open **`http://localhost:8000`** in your browser.

### 3. Run with Docker (Alternative)
```bash
docker build -t sonality .
docker run -p 8000:8000 sonality
```

### 4. Explore the Jupyter Notebooks (Optional)
```bash
jupyter notebook notebooks/01_cleaning.ipynb
```

---

## 🛠️ Tech Stack

* **Language & Modeling:** Python 3.11+, Scikit-Learn (K-Means, PCA, PowerTransformer), NumPy, Pandas, Scipy
* **Model Serialization:** Joblib
* **Backend API:** FastAPI, Uvicorn, Pydantic, HTTPX
* **Frontend:** Vanilla ES6+, CSS Container Queries, HTML5 Canvas 2D Visualizers, `html-to-image`
* **Audio Integration:** Apple Music / iTunes Search API
* **Cloud & Containerization:** Docker, Google Cloud Run, Cloudflare

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
