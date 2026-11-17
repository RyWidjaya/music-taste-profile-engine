import os
import json
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from scipy.stats import percentileofscore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    TrackSearchResult,
    TasteProfileRequest,
    TasteProfileResponse,
)

# ---------------------------------------------------------------------------
# GLOBAL IN-MEMORY MODEL ARTIFACT STORE
# ---------------------------------------------------------------------------
# In production ML, we never load models or heavy CSVs inside request handlers.
# That would cause disk reads on every HTTP request, blowing latency up to 500ms+.
# Instead, we load them ONCE into RAM at server startup (<1ms request latency).
# ---------------------------------------------------------------------------
artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Runs once when the server boots up, and cleans up when the server shuts down.
    """
    print("🚀 [Startup] Loading serialized ML models and dataset into RAM...")

    # Base paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "spotify_tracks_modeled.csv")
    models_dir = os.path.join(base_dir, "models")

    # 1. Load the fully modeled 89,583-track dataset into memory
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Modeled data not found at {data_path}. Run ml_pipeline.ipynb first.")
    artifacts["data"] = pd.read_csv(data_path)
    print(f" Loaded dataset: {len(artifacts['data']):,} tracks")

    # 2. Load metadata (persona names, features)
    meta_path = os.path.join(models_dir, "persona_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)
        print(" Loaded persona metadata")

    # 3. Load Scikit-Learn frozen models
    artifacts["kmeans"] = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    artifacts["power_transformer"] = joblib.load(os.path.join(models_dir, "power_transformer.joblib"))
    artifacts["pca_intensity"] = joblib.load(os.path.join(models_dir, "pca_intensity.joblib"))
    artifacts["pca_2d"] = joblib.load(os.path.join(models_dir, "pca_2d.joblib"))
    print(" Loaded all 4 joblib model artifacts")

    print(" System ready to serve predictions!")
    yield
    print("🛑 [Shutdown] Cleaning up resources...")
    artifacts.clear()


# ---------------------------------------------------------------------------
# FASTAPI APP INITIALIZATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Spotify Taste Profile Engine API",
    description="Production ML API that clusters songs into 7 Sound Personas and computes listener taste diagnostics.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) MIDDLEWARE
# ---------------------------------------------------------------------------
# Crucial for Full-Stack ML: Allows your frontend (e.g. localhost:3000, 5173,
# or your deployed web domain) to make fetch requests to this backend without
# browser security policy blocking them.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["System"])
def health_check():
    """Confirms the API server and ML models are loaded and healthy."""
    return {
        "status": "healthy",
        "loaded_tracks": len(artifacts.get("data", [])),
        "models_ready": "kmeans" in artifacts,
    }


# ---------------------------------------------------------------------------
# ENDPOINT 1: SONG AUTOCOMPLETE SEARCH
# ---------------------------------------------------------------------------
@app.get(
    "/api/search",
    response_model=list[TrackSearchResult],
    tags=["Search"],
    summary="Search tracks by title or artist",
)
def search_tracks(
    q: str = Query(..., min_length=1, description="Search keyword for track name or artist"),
    limit: int = Query(10, ge=1, le=50, description="Max number of results to return"),
):
    """
    Lightning-fast autocomplete search against our 89.5k dataset.
    Uses vectorized string matching across track_name and artists.
    """
    data = artifacts["data"]
    keyword = q.strip().lower()

    # Vectorized case-insensitive search across track name and artist
    mask = (
        data["track_name"].str.lower().str.contains(keyword, na=False)
        | data["artists"].str.lower().str.contains(keyword, na=False)
    )

    # Sort matches by popularity so famous songs appear first
    matches = (
        data[mask]
        .sort_values(by="popularity", ascending=False)
        .head(limit)
    )

    search_cols = ["track_id", "track_name", "artists", "album_name", "track_genre", "persona"]
    return matches[search_cols].to_dict(orient="records")


# ---------------------------------------------------------------------------
# ENDPOINT 2: TASTE PROFILE ENGINE
# ---------------------------------------------------------------------------
@app.post(
    "/api/taste-profile",
    response_model=TasteProfileResponse,
    tags=["ML Inference"],
    summary="Compute listener taste diagnostic from track IDs",
)
def get_taste_profile(request: TasteProfileRequest):
    """
    Production wrapper around our ML taste_profile() engine.
    Receives selected track_ids, computes persona breakdown, Shannon entropy,
    global percentiles, within-cluster recommendations, and 2D sound map points.
    """
    data = artifacts["data"]
    track_ids = request.track_ids

    # 1. Filter dataset for user's selected tracks
    selected = data[data["track_id"].isin(track_ids)].copy()

    if selected.empty:
        raise HTTPException(
            status_code=404,
            detail="None of the provided track IDs were found in the dataset.",
        )

    # 2. Persona Breakdown - % Breakdown of each Cluster
    breakdown = selected["persona"].value_counts(normalize=True).to_dict()

    # 3. Diversity Score - Normalized Shannon Entropy
    diversity_score = 0.0
    for prob in breakdown.values():
        if prob > 0:
            diversity_score += (-prob) * np.log(prob)
    diversity_score /= np.log(7)  # Normalize by log(7) (maximum possible entropy for 7 personas)

    # 4. Global Percentile Comparison
    percentiles = {}
    compare_features = ["danceability", "energy", "valence", "acousticness"]
    for feat in compare_features:
        user_avg = selected[feat].mean()
        percentiles[feat] = round(float(percentileofscore(data[feat], user_avg)), 1)

    # 5. Recommendations - Top 3 within-cluster nearest neighbors
    top_cluster = selected["cluster"].mode()[0]
    top_cluster_features = selected[selected["cluster"] == top_cluster]
    mean_point = (
        top_cluster_features["pca_x"].mean(),
        top_cluster_features["pca_y"].mean(),
    )

    # Filter other songs in same cluster not in user's selection
    other_songs = data[
        (data["cluster"] == top_cluster) & (~data["track_id"].isin(track_ids))
    ].copy()

    # Euclidean distance in 2D PCA space
    other_songs["dist"] = (
        (other_songs["pca_x"] - mean_point[0]) ** 2
        + (other_songs["pca_y"] - mean_point[1]) ** 2
    )

    cols = ["track_id", "track_name", "artists", "album_name"]
    suggested_songs = (
        other_songs.sort_values(by="dist", ascending=True)[cols]
        .head(3)
        .to_dict(orient="records")
    )

    # 6. 2D Sound Map Coordinates for User's Songs
    user_songs = selected[
        ["track_id", "track_name", "artists", "pca_x", "pca_y", "persona"]
    ].to_dict(orient="records")

    return {
        "persona_breakdown": breakdown,
        "diversity_score": round(float(diversity_score), 2),
        "percentiles": percentiles,
        "recommendations": suggested_songs,
        "user_songs": user_songs,
    }
