# Spotify Taste Profile Engine
### *Unsupervised Music Clustering & Full-Stack Inference Pipeline*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend%20API-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end applied machine learning system that clusters **89,500+ Spotify tracks** into data-driven **Sound Personas** using unsupervised learning, then delivers personalized listener taste breakdowns, diversity scores, population percentiles, and within-cluster recommendations.

---

## 📌 Why This Project?

Most music recommendation projects build supervised classifiers to predict existing `track_genre` labels (e.g., predicting "Rock" vs. "Pop"). That is simply classification wearing a clustering costume.

This project takes an **unsupervised approach**: rather than relying on noisy, human-assigned genre tags, it lets **$k$-Means discover its own sound categories** directly from continuous acoustic features. The result is a system that evaluates a real listener's taste profile independent of canned marketing labels.

```
                  ┌─────────────────────────────────────────┐
                  │          Raw Spotify Audio Data         │
                  │        (89,583 unique track rows)       │
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

### 1. Data Cleaning & Representation Bias Removal
* The raw 114k Spotify dataset contained ~24,000 duplicate tracks due to cross-genre tagging (e.g., the same song labeled as both "pop" and "dance").
* Retaining duplicates would have silently pulled cluster centroids toward over-represented commercial tracks. 
* Deduplicated the dataset strictly down to **89,583 unique songs**.

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
| **0** | **Upbeat Dance & Sunny Pop** | High danceability (+0.74), High valence (+0.89) | Reggaeton, Salsa, Disco, Latin Pop |
| **1** | **Chill Acoustic & Moody Ballads** | Low intensity (-1.29), Low valence (-0.45) | Acoustic, Folk, Singer-Songwriter, Blues |
| **2** | **Raw Energy & High-Octane Rock** | High liveness (+1.59), High intensity (+0.56) | Hard Rock, Punk, Live Performances |
| **3** | **Rhythmic Flow & Hip-Hop Beats** | High speechiness (+1.83), High danceability (+0.69) | Hip-Hop, Rap, R&B, Dancehall |
| **4** | **Hypnotic Beats & Electronic Club** | High instrumentalness (+1.44), High tempo (+0.22) | Techno, House, EDM, Trance |
| **5** | **Deep Ambient & Sleep Drift** | Ultra-low intensity (-1.89), High instrumentalness (+1.26) | Ambient, Classical, Sleep, New-Age |
| **6** | **Alternative Edge & Dark Grunge** | High intensity (+0.89), Low danceability (-0.79) | Grunge, Metal, Alternative Rock, Punk-Rock |

---

## ⚡ The Inference Engine (`taste_profile`)

The core inference engine receives a list of Spotify `track_id`s and computes:

1. **Persona Breakdown**: Normalized percentage distribution of the user's tracks across all 7 sound personas (feeds the interactive Donut Chart).
2. **Diversity Score (Shannon Entropy)**: Computes information entropy normalized against maximum entropy $\log(7)$:
   $$H = -\frac{\sum p_i \ln(p_i)}{\ln(7)}$$
   * `0.0` = hyper-focused listener (100% in one persona).
   * `1.0` = perfectly eclectic listener (equal split across all 7 personas).
3. **Comparative Percentiles**: Benchmarks user audio traits (*danceability, energy, valence, acousticness*) against the 89k-track population using `scipy.stats.percentileofscore`.
4. **Within-Cluster Recommendations**: Finds the centroid of the user's most common persona and retrieves the top 3 nearest-neighbor tracks (smallest Euclidean distance) not already in their library.
5. **2D Sound Map Coordinates**: Emits $(x, y)$ PCA coordinates for each song to plot them directly on the global interactive sound map.

---

## 📂 Repository Structure

```
├── data/
│   ├── spotify_tracks_clean.csv       # 89,583 cleaned, deduplicated tracks
│   └── spotify_tracks_modeled.csv     # Modeled tracks with cluster, persona & 2D PCA
├── models/
│   ├── kmeans_model.joblib            # Trained 7-cluster KMeans model
│   ├── power_transformer.joblib       # Fitted Yeo-Johnson transformer
│   ├── pca_intensity.joblib           # 1D PCA intensity triad model
│   ├── pca_2d.joblib                  # 2D PCA projection model
│   └── persona_metadata.json          # Persona names and feature schemas
├── backend/                           # FastAPI REST API (Tier 2)
│   ├── main.py                        # App entry point, lifespan loader & routes
│   └── schemas.py                     # Pydantic request/response schemas
├── 01_cleaning.ipynb                  # Data inspection & deduplication
├── 02_eda.ipynb                       # Skew analysis & feature correlations
├── ml_pipeline.ipynb                  # Transformation, clustering & taste engine
├── full_stack_ml_guide.md             # Architectural & engineering design guide
├── project_plan.md                    # Step-by-step milestone specification
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart

### Prerequisites
* Python 3.10+
* Virtual environment (`venv` recommended)

### 1. Clone & Setup
```bash
git clone https://github.com/RyWidjaya/spotify-taste-profile-engine.git
cd spotify-taste-profile-engine
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebooks
Launch Jupyter Notebook to explore the model training and evaluation:
```bash
jupyter notebook ml_pipeline.ipynb
```

---

## 🛠️ Tech Stack
* **Language & Core:** Python, NumPy, Pandas, Scipy
* **Machine Learning & Modeling:** Scikit-Learn (KMeans, PCA, PowerTransformer)
* **Model Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML5, TailwindCSS, Chart.js / Canvas

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
