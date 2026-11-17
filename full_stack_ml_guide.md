# Full-Stack Machine Learning Architecture Guide

A practical, intuition-first reference for building end-to-end applications that integrate Machine Learning models into production web services.

---

## 1. What is "Full-Stack Machine Learning"?

In traditional data science, work often ends inside a Jupyter Notebook: you clean data, fit a model, plot a graph, and print an evaluation metric. 

In **Full-Stack Machine Learning**, the notebook is only the research laboratory. The goal is to take that model, package it as a robust software component, serve it through a high-performance API, and present it to an end user through an intuitive, interactive interface.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE FULL-STACK ML LIFECYCLE                           │
│                                                                             │
│   [ Raw Data ]                                                              │
│        │                                                                    │
│        ▼ (Tier 1: ML Pipeline)                                              │
│   [ Cleaning & EDA ] ──► [ Feature Pipeline ] ──► [ Model Training ]        │
│                                                          │                  │
│                                                          ▼                  │
│                                                 [ Serialized Artifacts ]    │
│                                                  (.joblib / .pkl)           │
│                                                          │                  │
│        ┌─────────────────────────────────────────────────┘                  │
│        ▼ (Tier 2: Backend API)                                              │
│   [ FastAPI Server ] ◄── In-Memory Model Cache (Loaded at Startup)          │
│        │                                                                    │
│        │  HTTP POST /api/taste-profile                                      │
│        │  (Pydantic Validation, Vector Transformation, Cluster Inference)   │
│        │                                                                    │
│        ▼ (Tier 3: Frontend UI)                                              │
│   [ Web Application ] ──► Charts, Spotify Players, Interactive Taste Cards   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three-Tier Architecture

To make the codebase maintainable, scalable, and easy to debug, we strictly decouple the application into three independent tiers. Each tier only needs to know the **input and output contract** of the tier next to it.

```
+-------------------------------------------------------------------------+
| Tier 1: The ML Engine (Offline / Pipeline)                              |
| Python | Pandas | Scikit-learn | Joblib                                 |
| - Runs during development / offline training                            |
| - Produces: Clean data, scaler.joblib, kmeans.joblib, pca.joblib        |
| - Verifiable with zero web code (notebook / standalone script)          |
+------------------------------------+------------------------------------+
                                     | Loads artifacts at startup
                                     v
+-------------------------------------------------------------------------+
| Tier 2: The Backend API (The Bridge)                                    |
| FastAPI | Uvicorn | Pydantic                                            |
| - Loads models into RAM ONCE when the server boots                      |
| - Receives structured JSON requests from the frontend                   |
| - Runs lightning-fast in-memory inference                               |
| - Returns structured JSON responses (scores, percentages, metrics)      |
+------------------------------------+------------------------------------+
                                     | Sends fetch() / receives JSON
                                     v
+-------------------------------------------------------------------------+
| Tier 3: The Frontend (The Stage)                                        |
| HTML / Modern JS (or React) | Tailwind CSS | Chart.js                   |
| - User interactions: song search, autocomplete, selection tags          |
| - Data visualization: persona donut charts, 2D PCA scatter map          |
| - Multimedia: Spotify embedded audio players                            |
| - "Vibecoded" freely against the API's JSON contract                    |
+-------------------------------------------------------------------------+
```

---

## 3. Tier 1: The ML Engine & Modeling Pipeline

### Key Technologies
- **Pandas / NumPy**: Ingestion, vector math, data manipulation.
- **Scikit-learn**: Feature scaling (`StandardScaler` or `PowerTransformer`), clustering (`KMeans`), dimensionality reduction (`PCA`), distance metrics (`cosine_similarity`).
- **SciPy**: Statistical percentiles (`scipy.stats.percentileofscore`), Shannon entropy.
- **Joblib**: Efficient serialization of NumPy arrays and Scikit-learn estimators.

### The Pipeline Steps
1. **Ingestion & Deduplication**: Ensuring training data is free of bias (e.g. repeated track IDs across multiple genres).
2. **Feature Engineering & Transformation**:
   - Selecting purely acoustic features (ignoring metadata like track title or popularity).
   - Correcting scale and skew (e.g. `PowerTransformer` / `StandardScaler`).
3. **Unsupervised Modeling**:
   - Choosing optimal $k$ via Elbow Method (inertia) and Silhouette Analysis.
   - Fitting `KMeans(n_clusters=k)`.
   - Dimension reduction via `PCA(n_components=2)` to create a 2D coordinate space for plotting.
4. **Cluster Interpretation (Z-Scoring)**:
   - Calculating cluster feature averages relative to global means:
     $$z = \frac{\mu_{\text{cluster}} - \mu_{\text{global}}}{\sigma_{\text{global}}}$$
   - Assigning human-interpretable persona names (e.g., *"Late Night Acoustic"*, *"High-Energy Hype"*).
5. **Serialization**: Exporting the fitted objects:
   - `scaler.joblib`
   - `kmeans_model.joblib`
   - `pca_model.joblib`
   - `track_database.parquet` (or `.csv` for fast lookups)

> [!IMPORTANT]
> **Golden Rule of ML Serving**: Never refit a model or scaler on user input during live serving. New user tracks are transformed using the **pre-fitted** `.transform()` method and predicted using `.predict()`.

---

## 4. Tier 2: The Serving Layer (Backend API)

### Key Technologies
- **FastAPI**: Modern, high-performance web framework for building APIs with Python.
- **Uvicorn**: Lightning-fast ASGI server implementation to run FastAPI.
- **Pydantic**: Data validation and type safety using Python type annotations.

### Why FastAPI for Machine Learning?
1. **Speed & Async**: Built on Starlette and Pydantic, nearly as fast as NodeJS and Go.
2. **Automatic Data Validation**: Rejects malformed requests before they touch the ML model.
3. **In-Memory Model Caching**: Loads heavy model files (`.joblib`) into memory **once** on application boot.
4. **Auto-Generated Documentation**: Instant interactive API docs via Swagger UI (`/docs`).

### Recommended API Contract

#### Request: `POST /api/taste-profile`
```json
{
  "track_ids": [
    "5SuOikwiRyPMVoIQDJUgSV",
    "4qPNDBW1i3p13qLCt0Ki3A",
    "1iJBSr7s7jYXzM8EGcbK5b"
  ]
}
```

#### Response: `200 OK`
```json
{
  "user_summary": {
    "total_tracks": 15,
    "top_persona": "Chill Acoustic",
    "entropy_score": 1.42,
    "diversity_label": "Eclectic Listener"
  },
  "persona_breakdown": [
    { "persona": "Chill Acoustic", "percentage": 45.0, "color": "#1DB954" },
    { "persona": "High-Energy Pop", "percentage": 35.0, "color": "#3B82F6" },
    { "persona": "Moody Instrumental", "percentage": 20.0, "color": "#8B5CF6" }
  ],
  "percentile_comparisons": {
    "acousticness": 84.5,
    "energy": 32.1,
    "danceability": 68.0
  },
  "pca_coordinates": [
    { "track_id": "5SuOikwiRyPMVoIQDJUgSV", "name": "Comedy", "x": 1.24, "y": -0.85, "cluster": 0 }
  ],
  "recommendations": [
    {
      "track_id": "2x0RZdkZcD8tHn5298slpP",
      "track_name": "Autumn Breeze",
      "artists": "Luna Nova",
      "persona": "Chill Acoustic",
      "similarity_score": 0.94
    }
  ]
}
```

---

## 5. Tier 3: The Presentation Layer (Frontend UI)

### Philosophy: "Vibecoding" with Structure
The frontend is the visual showcase. Because you are training for ML Engineering rather than UI development, this layer is designed to be **vibecoded**: generating and rapidly iterating with modern tools, styling frameworks, and AI code generation.

The only prerequisite for effortless frontend vibecoding is having a **predictable backend response schema** (the JSON contract above).

### Core UI Modules
1. **Search & Selection Tray**:
   - Autocomplete search bar querying `/api/search?q={query}`.
   - Selected track tags with "remove" buttons.
   - "Analyze My Sound" trigger button.
2. **Taste Breakdown Dashboard**:
   - Donut Chart showing persona percentages (using Chart.js or Recharts).
   - Taste Diversity / Entropy Gauge.
   - Percentile call-out badges (e.g. *"Your acousticness is higher than 84% of listeners"*).
3. **Interactive Sound Map**:
   - 2D Scatter plot of the 89K song universe in background, with the user's selected songs plotted as glowing pins.
4. **Recommended Tracks**:
   - Cards displaying recommended songs with native Spotify embedded iframes:
     ```html
     <iframe src="https://open.spotify.com/embed/track/{track_id}" 
             width="100%" height="80" frameBorder="0" allow="encrypted-media"></iframe>
     ```

---

## 6. End-to-End Request Journey

Here is the exact step-by-step path of data through the entire system:

```
[ User Action ]
  │ User selects 10 songs on the webpage and clicks "Analyze".
  ▼
[ Frontend: HTTP Request ]
  │ JS collects track IDs: ['id1', 'id2', ...]
  │ Sends POST request with JSON payload to https://api.../api/taste-profile
  ▼
[ Backend: Request Ingestion & Validation ]
  │ FastAPI / Pydantic verifies the request body has valid track IDs.
  │ Invalid IDs or empty payloads return an immediate 422 Unprocessable Entity.
  ▼
[ Backend: Database & Feature Lookup ]
  │ Fast in-memory lookup retrieves the 9 raw audio features for those 10 tracks.
  ▼
[ Backend: ML Transformation & Prediction ]
  │ scaler.transform(features)  ──► normalized 9D feature vectors
  │ kmeans.predict(vectors)     ──► cluster assignments for each song
  │ pca.transform(vectors)      ──► 2D coordinates for the UI scatter plot
  ▼
[ Backend: Taste Metrics Engine ]
  │ Percentage breakdown calculated per persona.
  │ Shannon entropy computed for listening diversity score.
  │ scipy percentile ranking calculated against the 89K population baseline.
  │ Nearest neighbors found via cosine similarity for recommended songs.
  ▼
[ Backend: HTTP Response ]
  │ Serializes all metrics into clean JSON and returns HTTP 200.
  ▼
[ Frontend: Rendering & Visualization ]
  │ Chart.js updates the donut chart.
  │ SVG scatter plot draws user songs on the PCA map.
  │ Recommendation cards populate with playable Spotify player embeds.
```

---

## 7. Standard Project Directory Structure

As we transition from raw research notebooks to a production-ready repository, this is the industry-standard structure we will build towards:

```
KMeans-Music-Taste-Full-Stack/
├── data/
│   ├── spotify_tracks_raw.csv        # Original raw data
│   └── spotify_tracks_clean.csv      # Cleaned, deduplicated data (89K rows)
│
├── notebooks/
│   ├── 01_cleaning.ipynb             # Phase 1: Data audit & cleaning
│   ├── 02_eda.ipynb                  # Phase 1: Feature distributions & correlations
│   ├── 03_clustering_and_eval.ipynb  # Phases 2-5: k sweep, PCA, persona naming
│   └── 04_engine_prototype.ipynb     # Phase 6: taste_profile() notebook prototype
│
├── models/                           # Saved artifacts (joblib)
│   ├── scaler.joblib                 # Preprocessing scaler / transformer
│   ├── kmeans_model.joblib           # Fitted KMeans clustering model
│   ├── pca_2d.joblib                 # Fitted 2D PCA projector
│   └── persona_metadata.json         # Persona names, colors, and definitions
│
├── backend/                          # Tier 2: FastAPI application
│   ├── main.py                       # App entrypoint, CORS, lifespan startup
│   ├── schemas.py                    # Pydantic request/response models
│   ├── services/
│   │   ├── recommender.py            # Cosine-similarity nearest neighbors
│   │   └── taste_engine.py           # Breakdown %, entropy, percentiles
│   └── requirements.txt              # FastAPI, uvicorn, scikit-learn, joblib
│
├── frontend/                         # Tier 3: Web UI
│   ├── index.html                    # Main application HTML
│   ├── css/
│   │   └── style.css                 # Tailwind / custom styling
│   └── js/
│       ├── api.js                    # fetch() calls to FastAPI endpoints
│       ├── charts.js                 # Chart.js donut and PCA visualizers
│       └── app.js                    # UI state and event handlers
│
├── project_plan.md                   # Overall project roadmap
├── full_stack_ml_guide.md            # This reference guide
└── README.md                         # Portfolio project showcase
```

---

## 8. Five Classic Traps in Full-Stack ML (And How to Avoid Them)

| Trap | Why it happens | The Solution |
| :--- | :--- | :--- |
| **1. Training-Serving Skew** | Preprocessing in the API is coded differently than in the training notebook. | Always export the fitted `StandardScaler` / `PowerTransformer` as a `.joblib` object and call `.transform()` in the API. |
| **2. Cold Inference Latency** | Loading the `.joblib` model inside the request function (`def endpoint(...)`), reading disk on every request. | Load models once at startup using FastAPI's `@asynccontextmanager` lifespan event into global or app state. |
| **3. Giant Response Payloads** | Sending all 89,000 track coordinates to the frontend on every request. | Pre-generate downsampled background points once or serve them via static asset / CDN. |
| **4. CORS Blocking** | Frontend (`localhost:3000`) tries to call Backend (`localhost:8000`) and the browser blocks it. | Add `CORSMiddleware` in FastAPI with explicit allowed origins. |
| **5. Notebook Coupling** | Trying to `import` functions directly from a `.ipynb` file into the API. | Clean ML logic into modular Python scripts (`.py`) before building the backend. |

---

*Keep this document open as a north star while we progress through each phase of the project.*
