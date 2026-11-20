# Music Taste Profile Engine
### *Unsupervised Music Clustering & Full-Stack Inference Pipeline*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend%20API-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end applied machine learning system that clusters **127,900+ Spotify tracks (spanning up through 2026)** into data-driven **Sound Personas** using unsupervised learning, then delivers personalized listener taste breakdowns, diversity scores, population percentiles, and within-cluster recommendations.

---

## 📌 Why This Project?

Most music recommendation projects build supervised classifiers to predict existing `track_genre` labels (e.g., predicting "Rock" vs. "Pop"). That is simply classification wearing a clustering costume.

This project takes an **unsupervised approach**: rather than relying on noisy, human-assigned genre tags, it lets **$k$-Means discover its own sound categories** directly from continuous acoustic features. The result is a system that evaluates a real listener's taste profile independent of canned marketing labels.

```
                  ┌─────────────────────────────────────────┐
                  │    Spotify Audio Data (2024 - 2026)     │
                  │       (127,900 unique track rows)       │
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
                  │        Interactive Web Dashboard        │
                  │  (Donut Chart, Entropy Badge, Sound Map)│
                  └─────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline & Engineering Decisions

### 1. Data Cleaning & Catalog Augmentation (2024–2026 Extension)
* **Deduplication of Baseline Data**: The raw 114k Spotify dataset contained ~24,000 duplicate tracks due to cross-genre tagging (e.g., the same song labeled as both "pop" and "dance"). Retaining duplicates would have silently pulled cluster centroids toward over-represented commercial tracks. Cleaned strictly down to **89,583 unique baseline songs**.
* **Incremental Ingestion & Recency Augmentation**: The original dataset only spanned through 2024 and lacked major recent releases as well as comprehensive Gen 3–5 K-Pop representation. To modernize the catalog, we extracted recent high-popularity releases from a 56M-row Spotify warehouse dataset (`spotify-huge-audio-features.parquet`).
* **Noise Filtering & Schema Preservation**:
  * Uncurated scraping would ingest millions of obscure/spam uploads; applied popularity thresholds (`popularity >= 18` for curated K-Pop artists, `popularity >= 50` for general 2025–2026 releases).
  * Implemented exact-artist matching to prevent false-positive genre contamination (e.g., *Bibi Blocksberg* radio plays or *Dean Lewis* pop tracks accidentally tagged as K-Pop).
  * Handled the late-2024 deprecation of Spotify's public Audio Features API by curating major 2026 comebacks (*Hearts2Hearts – Lemon Tang*, *NMIXX – Heavy Serenade*, *BABYMONSTER – DESERVE / CHOOM*, *aespa – Lemonade*, *IVE – Blackhole*, etc.) with acoustic profiling.
  * Reached a clean, production-ready dataset of **127,900 unique songs** with 0 nulls.

### 2. Distribution Correction (Yeo-Johnson)
* Features like `speechiness`, `instrumentalness`, and `liveness` exhibit heavy positive skew and zero-inflation.
* Replaced standard linear scaling with the **Yeo-Johnson PowerTransformer** to stabilize variance and normalize feature distributions without clipping or data loss.

### 3. Collinearity Resolution (1D PCA Intensity Triad)
* Exploratory data analysis revealed high pairwise collinearity between `energy`, `loudness`, and `acousticness` ($|r| > 0.70$).
* Clustered on these separately would cause $k$-Means distance metrics to over-index on loudness/volume at the expense of rhythmic and structural features.
* Applied **Principal Component Analysis (PCA)** to collapse this triad into a single composite feature: **`intensity`** (capturing ~78% of shared variance).

### 4. Model Selection: Why $k = 7$?
* Evaluated clusters across $k \in [2, 10]$ on a 10,000-sample slice using **Elbow inertia** and **Silhouette analysis**.
* While $k=5$ yielded a marginally higher silhouette score ($0.173$ vs $0.166$), inspecting cluster composition revealed severe semantic over-compression: ambient sleep tracks were lumped with acoustic ballads, and techno was grouped with death metal.
* **$k = 7$ was chosen** because it cleanly separated distinct acoustic genres while retaining statistical coherence and high domain interpretability.

---

## 🎧 The 7 Universal Sound Personas

By analyzing standardized Z-scores across audio features, corroborating with top genre tags, and evaluating sample tracks, the 7 clusters were mapped to universal sound personas:

| Cluster | Sound Persona | Distinguishing Acoustic Signature | Common Genres / Representative Tracks |
| :---: | :--- | :--- | :--- |
| **0** | **Heavy & Live** | Ultra-high intensity (+1.55), High liveness (+0.70), High tempo (+0.57) | Metalcore, Heavy Metal, Hard Rock, Punk |
| **1** | **Driven Alt & Punk** | High tempo (+0.80), Low intensity (-0.98), Low speechiness (-0.66) | Alternative, Indie Rock, Dark Wave, Cantopop |
| **2** | **Synth & Club** | Ultra-high instrumentalness (+1.87), High intensity (+0.86) | Techno, House, EDM, Breakbeat, Trance |
| **3** | **Vocal & Rhythmic** | High speechiness (+1.52), High danceability (+0.83) | Hip-Hop, Rap, Dancehall, Trap |
| **4** | **Bright Pop & Groove** | High valence (+0.83), High danceability (+0.69), High intensity (+0.69) | Dance-Pop, Disco, Salsa, Forro, Upbeat K-Pop |
| **5** | **Unplugged & Intimate** | Low intensity (-1.31), Low tempo (-1.02), Low speechiness (-0.65) | Acoustic, Folk, Singer-Songwriter, Romance, Ballads |
| **6** | **Ambient & Drone** | Ultra-low intensity (-2.70), High instrumentalness (+1.90), Low valence (-1.09) | Ambient, Classical, Sleep, New-Age, Meditation |

---

## ⚡ The Inference Engine (`taste_profile`)

The core inference engine receives a list of Spotify `track_id`s and computes:

1. **Persona Breakdown**: Normalized percentage distribution of the user's tracks across all 7 sound personas (feeds the interactive Donut Chart).
2. **Diversity Score (Shannon Entropy)**: Computes information entropy normalized against maximum entropy $\log(7)$:
   $$H = -\frac{\sum p_i \ln(p_i)}{\ln(7)}$$
   * `0.0` = hyper-focused listener (100% in one persona).
   * `1.0` = perfectly eclectic listener (equal split across all 7 personas).
3. **Comparative Percentiles**: Benchmarks user audio traits (*danceability, energy, valence, acousticness*) against the 128k-track population using `scipy.stats.percentileofscore`.
4. **Within-Cluster Recommendations**: Finds the centroid of the user's most common persona and retrieves the top 3 nearest-neighbor tracks (smallest Euclidean distance) not already in their library.
5. **2D Sound Map Coordinates**: Emits $(x, y)$ PCA coordinates for each song to plot them directly on the global interactive sound map.

---

## 📂 Repository Structure

```
├── data/
│   ├── spotify-tracks-dataset-detailed.csv # Baseline raw 114k Spotify dataset
│   ├── spotify_tracks_clean.csv            # Cleaned baseline dataset (89,583 tracks)
│   ├── spotify_master_2026.csv             # Augmented 2026 master dataset (127,900 tracks)
│   └── spotify_tracks_modeled.csv          # 127.9k modeled tracks with cluster, persona & 2D PCA
├── models/
│   ├── kmeans_model.joblib                 # Trained 7-cluster KMeans model (128k tracks)
│   ├── power_transformer.joblib            # Fitted Yeo-Johnson transformer
│   ├── pca_intensity.joblib                # 1D PCA intensity triad model (energy/loudness/acousticness)
│   ├── pca_2d.joblib                       # 2D PCA sound-map projection model
│   └── persona_metadata.json               # Persona names, cluster labels & feature schemas
├── backend/                                # FastAPI REST API (Tier 2)
│   ├── main.py                             # App entry point, lifespan loader & REST routes
│   └── schemas.py                          # Pydantic request/response schemas
├── 01_cleaning.ipynb                       # Step 1: Baseline data audit, deduplication & null handling
├── 02_eda.ipynb                            # Step 2: Exploratory data analysis, distributions & multicollinearity
├── 03_data_augmentation.ipynb              # Step 3: 2025-2026 Kaggle ingestion & curated title track synthesis
├── 04_ml_pipeline.ipynb                    # Step 4: Production Yeo-Johnson, 1D PCA, KMeans & taste engine
├── full_stack_ml_guide.md                  # Comprehensive full-stack ML architectural guide
├── project_plan.md                         # Milestone project roadmap & technical tasks
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart

### Prerequisites
* Python 3.10+
* Virtual environment (`venv` recommended)

### 1. Clone & Setup
```bash
git clone https://github.com/RyWidjaya/music-taste-profile-engine.git
cd music-taste-profile-engine
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
uvicorn backend.main:app --reload
```
Open `http://localhost:8000` to interact with the Sonality taste profiler.

### 3. Run the Notebooks (Optional)
Launch Jupyter Notebook to explore data preparation, augmentation, and model training:
```bash
# Sequential notebook workflow (01 -> 02 -> 03 -> 04):
jupyter notebook notebooks/01_cleaning.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_data_augmentation.ipynb
jupyter notebook notebooks/04_ml_pipeline.ipynb
```

---

## 🛠️ Tech Stack
* **Language & Core:** Python 3.11+, NumPy, Pandas, Scipy
* **Machine Learning & Modeling:** Scikit-Learn (KMeans, PCA, PowerTransformer / Yeo-Johnson)
* **Model Serialization:** Joblib
* **Backend:** FastAPI, Uvicorn, Pydantic, HTTPX
* **Frontend:** Vanilla ES6+, Modern Responsive CSS (Container Queries, Dark Editorial Aesthetic), HTML5 Canvas 2D Visualizers, `html-to-image`
* **Containerization & Cloud:** Docker, Google Cloud Run / Render

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
