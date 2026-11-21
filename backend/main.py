"""
backend/main.py: The API Gateway & Serving Layer
==============================================================================
This file connects your Machine Learning pipeline to the outside web.
It is architected around the 4 Universal Pillars of production FastAPI services:

  PILLAR 1: Preparation  ──► Load models & data into RAM once (Lifespan)
  PILLAR 2: App Instance ──► Create central FastAPI app and attach metadata
  PILLAR 3: Middleware   ──► Grant browser security permissions (CORS)
  PILLAR 4: Routes       ──► Define URLs that validate input and call engine.py
==============================================================================
"""

import os
import re
import time
import json
import joblib
import pandas as pd
import requests
import urllib.parse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Data contracts (Pydantic schemas)
from backend.schemas import (
    TrackSearchResult,
    TasteProfileRequest,
    TasteProfileResponse,
)

# Pure ML inference math (zero web code)
from backend.engine import taste_profile


# ==============================================================================
# PILLAR 1: IN-MEMORY ARTIFACT STORE & LIFESPAN LOADER (Preparation)
# ==============================================================================
# In production ML, we NEVER load models or heavy CSVs inside route functions.
# Reading disk on every HTTP request adds 300ms+ of lag and risks memory spikes.
# Instead, we load models ONCE into RAM at boot so requests answer in < 5ms.
# ==============================================================================

# Global in-memory warehouse accessible by all route functions
# Once populated during startup, every route function inside can instantly access
# artifacts["data"] or artifacts["kmeans"] directly from RAM without reloading 
artifacts = {}

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@asynccontextmanager # used to split server execution into two distinct phases separated by the yield keyword
async def lifespan(app: FastAPI):
    """
    Reads heavy models and data from disk ONCE at server startup and stores them
    in the global `artifacts` dict, so routes can access them instantly from RAM
    instead of repeatedly calling pd.read_csv() or joblib.load() on every request.

    Lifespan Context Manager: Controls the server's lifecycle.
      - Code BEFORE `yield`: Executes ONCE at server boot (Startup).
      - The `yield`: The server pauses here and answers user requests.
      - Code AFTER `yield`: Executes ONCE when server shuts down (Teardown).
    """

    # Construct OS-independent paths relative to project root
    # Creates robust full (absolute) paths dynamically at runtime by combining the relative pieces we have
    base_dir = BASE_DIR # returns the full path to the project's root folder (to construct full paths from relative pieces)
    data_path = os.path.join(base_dir, "data", "spotify_tracks_modeled.csv")
    models_dir = os.path.join(base_dir, "models")

    # Load the modeled 127,900-track dataset into RAM
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Modeled data not found at {data_path}. Run 04_ml_pipeline.ipynb first.")
    
    # instead of calling pd.read_csv every single time it's read once when the server starts and stored in a dict
    artifacts["data"] = pd.read_csv(data_path) # store dataset to artifacts dict for easy access - lives in RAM 

    # Load persona metadata definitions
    meta_path = os.path.join(models_dir, "persona_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)

    # Load trained Scikit-Learn transformers and KMeans clusterer into RAM
    artifacts["kmeans"] = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    artifacts["power_transformer"] = joblib.load(os.path.join(models_dir, "power_transformer.joblib"))
    artifacts["pca_intensity"] = joblib.load(os.path.join(models_dir, "pca_intensity.joblib"))
    artifacts["pca_2d"] = joblib.load(os.path.join(models_dir, "pca_2d.joblib"))


    # ─── 2. THE SERVING PAUSE ─────────────────────────────────────────────────
    yield  # The server stays awake here, processing incoming user requests!
           # yield pauses the function and holds its state 
           # When yield is hit, FastAPI knows startup preparation is complete 
           # FastAPI opens port 8000 and starts accepting HTTP traffic 
           # The lifespan function hangs in suspended animation at this exact 
           # line for as long as the server runs 

    # ─── 3. SHUTDOWN CLEANUP ──────────────────────────────────────────────────
    # When the server has finished running (server is shutting down) the dict 
    # is cleared (RAM is cleared) and the service ends 
    artifacts.clear()


# ==============================================================================
# PILLAR 2: APPLICATION INSTANTIATION (The Central App Object)
# ==============================================================================
# Creates the central `app` object that manages routes, settings, and docs.
# The `lifespan=lifespan` argument binds our Pillar 1 startup logic to this app.
# ==============================================================================
app = FastAPI(
    title="Sonality API",
    description="Production ML API that clusters songs into 7 Sound Personas and computes listener taste diagnostics.",
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# PILLAR 3: CORS SECURITY MIDDLEWARE (Browser Access Handshake)
# ==============================================================================
# Web browsers enforce the 'Same-Origin Policy': they block frontends 
# (e.g. localhost:3000 or vercel.app) from fetching data from backends on another port
# (e.g. localhost:8000 or render.com) unless the API explicitly sends permission.
# ==============================================================================
app.add_middleware( # .add_middleware() adds functionality that intercepts/processes everything travelling between the outside world and your routes
    CORSMiddleware,            # Intercepts every request to handle browser Cross-Origin security checks
    allow_origins=["*"],       # Whitelist of allowed frontend domains ("*" allows all; in prod use ["https://your-site.vercel.app"])
    allow_credentials=True,    # Allows frontend to pass cookies, authorization headers, or session tokens across origins
    allow_methods=["*"],       # Allowed HTTP actions ("*" permits GET, POST, OPTIONS, PUT, DELETE, etc.)
    allow_headers=["*"],       # Allowed HTTP headers sent by browser ("*" permits Content-Type, Authorization, etc.)
)


# ==============================================================================
# PILLAR 4: THE ROUTES / ENDPOINTS (The Doors to the Outside World)
# ==============================================================================
# Golden Rule: Keep routes thin. Route functions should only:
#   1. Read pre-loaded data from `artifacts`.
#   2. Validate inputs.
#   3. Delegate heavy math to `engine.py`.
#   4. Return the result.
# ==============================================================================

# ─── ROUTE: SERVE FRONTEND (HTTP GET / and /static/*) ─────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.get("/", include_in_schema=False)
def serve_index():
    """Serves the single-page frontend application at the root URL."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)

# styles.css, app.js, favicon.svg and the logo files live next to index.html
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/robots.txt", include_in_schema=False)
def serve_robots():
    """Instructs Googlebot and search engine crawlers to index the site."""
    return PlainTextResponse("User-agent: *\nAllow: /\nSitemap: https://app.sonality.workers.dev/sitemap.xml\n", media_type="text/plain")

@app.get("/sitemap.xml", include_in_schema=False)
def serve_sitemap():
    """XML sitemap for Google Search Console and web crawlers."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://app.sonality.workers.dev/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>"""
    return Response(content=xml_content, media_type="application/xml")



# ─── ROUTE 0: SYSTEM HEALTH CHECK ─────────────────────────────────────────────
# Cloud hosts (Render, AWS, GCP) automatically ping this route every 10-30s
# to verify that your service is healthy and hasn't crashed.
# Checks the server is awake and that everything required is loaded in RAM

# Whenever a client makes an HTTP GET request to the path /api/health, run the function 
@app.get("/api/health", tags=["System"]) 
def health_check():
    """Confirms the API server and ML models are loaded in RAM and healthy."""
    return {
        "status": "healthy",
        "loaded_tracks": len(artifacts.get("data", [])), # checks how many tracks are currently loaded in RAM
        "models_ready": "kmeans" in artifacts, # returns True if the trained KMeans model is ready in RAM 
    }


# ─── ROUTE 1: AUTOCOMPLETE SONG SEARCH (HTTP GET) ─────────────────────────────
# We use GET because the client is just LOOKING UP data with a URL query string.
# A GET request carries its input in the URL (the query string), and has no body.
# e.g. input from the client is carried in a portion of the url (the query string)

# Triggered by: GET /api/search?q=queen&limit=10

# Flow: user types "queen" in the textbox 
#   -> frontend JS builds /api/search?q=queen&limit=10 
#   -> FastAPI calls search_tracks(q="queen", limit=10)
#   -> returned list is sent back as JSON and shown as dropdown results.
#
# q and limit are QUERY PARAMETERS (key=value pairs after the "?", split by "&").
# FastAPI matches them BY NAME to the function arguments, converts them to the
# type hint (str / int) and validates them. Bad input (missing q, limit=500)
# gets a 422 before this code runs.

# When a request is made to path /api/search run the function below 
# response_model indicates what shape the returned data must have - should be a list with the TrackSearchResult format (output contract from schemas)
# tags groups related endpoints into groups - e.g. tags=["Search"] puts /api/search in a "Search" folder 
# summary is the title displayed next to the URL in the /docs page 
@app.get("/api/search", response_model=list[TrackSearchResult], tags=["Search"], summary="Search tracks by title or artist",)
def search_tracks(
    # Query(...) indicates to read the value from the URL's query string (the "?q=queen&limit=10" part, as key=value pairs) and validates it against the rules (min_length, ge/le).
    q: str = Query(..., min_length=1, description="Search keyword for track name or artist"), # converts the query to a python string
    limit: int = Query(30, ge=1, le=100, description="Max number of results to return"),
):
    """
    Lightning-fast autocomplete search against our 128k in-memory dataset.
    Uses vectorized string matching across track_name and artists.
    """

    # obtains data and the keyword stripped 
    data = artifacts["data"] # data is the df from the csv 
    keyword = q.strip().lower()

    # Fast vectorized search in RAM (boolean masking) - either looking for matching track name or artist
    # Returns sequence of T/F for every single track indicating if it matches
    mask = (
        data["track_name"].str.lower().str.contains(keyword, na=False)
        | data["artists"].str.lower().str.contains(keyword, na=False)
    )

    # Sort matches by popularity descending so famous songs show up first
    # Indexes data using the boolean mask to return the matches and sorts them returning only "limit" number of matches
    matches = (
        data[mask]
        .sort_values(by="popularity", ascending=False)
        .head(limit)
    )

    # Return only the 6 columns required by TrackSearchResult
    # always return a dict so it can be rendered as json
    search_cols = ["track_id", "track_name", "artists", "album_name", "track_genre", "persona"]
    return matches[search_cols].to_dict(orient="records")


# ─── ROUTE 2: TASTE PROFILE ML INFERENCE (HTTP POST) ──────────────────────────
# We use POST because the client is SUBMITTING a structured JSON payload in the
# HTTP request body.
# - Input contract: `TasteProfileRequest` validates track_ids is a non-empty list.
# - Output contract: `TasteProfileResponse` guarantees the 5 diagnostic metrics.
@app.post("/api/taste-profile", response_model=TasteProfileResponse, tags=["ML Inference"], summary="Compute listener taste diagnostic from track IDs")
def get_taste_profile(request: TasteProfileRequest, background_tasks: BackgroundTasks): 
    # `request` is just the parameter name; `TasteProfileRequest` is its type/contract. 
    # Because it's a Pydantic schema, FastAPI reads the JSON body, validates it against the schema, and passes in the finished object.
    # What is passed is an object with the track_ids (instance of TasteProfileRequest class)
    """
    Production wrapper around our ML taste_profile() engine.
    Receives selected track_ids, validates them, and delegates pure computation
    to engine.py.
    """

    data = artifacts["data"]
    track_ids = request.track_ids # pulls the list of ids passed from the input object (check schemas.py for the object's structure)

    # Defensive check: ensure at least one requested track exists in our dataset
    selected = data[data["track_id"].isin(track_ids)]
    if selected.empty: # raise error if there are no matches (defensive check)
        raise HTTPException(status_code=404, detail="None of the provided track IDs were found in the dataset.")

    # Delegate the pure mathematical computation to backend/engine.py
    # engine.py is imported which contains the KMeans Algorithm function required which is called upon and its output is returned
    profile = taste_profile(track_ids, data)

    # Server-side background pre-warming: fetch previews for recommended songs ahead of time
    def prewarm_recs(recs):
        for r in recs:
            try:
                get_song_preview(track_name=r.get("track_name", ""), artist=r.get("artists", ""))
            except Exception:
                pass
    background_tasks.add_task(prewarm_recs, profile.get("recommendations", []))

    return profile


# ─── ROUTE 3: SAMPLED PCA POINTS FOR SOUND MAP (HTTP GET) ─────────────────────
@app.get("/api/sound-map", tags=["Visualisation"], summary="Sampled PCA points for the sound map")
def sound_map(n: int = Query(3500, ge=500, le=20000)):
    """
    Returns sampled 2D PCA points across all 127,900 tracks for background scatterplot
    rendering, with balanced stratified sampling across all 7 sound personas so dots
    appear cleanly and uniformly distributed without dense clumping.
    """
    key = f"sound_map_{n}"
    if key not in artifacts:
        import pandas as pd
        df = artifacts["data"]
        per_persona = max(50, n // 7)
        sampled = pd.concat([
            g.sample(n=min(len(g), per_persona), random_state=42)
            for _, g in df.groupby("persona")
        ])
        artifacts[key] = {
            "points": sampled[["pca_x", "pca_y", "persona"]].round(3).to_dict(orient="records"),
            "explained_variance": [round(float(v), 3) for v in artifacts["pca_2d"].explained_variance_ratio_[:2]],
        }
    return artifacts[key]


# ─── ROUTE 4: AUDIO CHORUS PREVIEW (HTTP GET) ─────────────────────────────────
preview_cache = {}
http_session = requests.Session()
http_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

def clean_music_query(track_name: str, artist: str):
    """
    Cleans track titles and collaborator lists to maximize search engine hit rate.
    Extracts primary artist (handles semicolons/commas) and strips remastered/live tags.
    """
    primary_artist = re.split(r"[;,]", artist)[0].strip()
    clean_title = re.sub(r"\(feat\..*?\)", "", track_name, flags=re.IGNORECASE)
    clean_title = re.sub(r"\[feat\..*?\]", "", clean_title, flags=re.IGNORECASE)
    clean_title = re.sub(r"-\s*(remaster(ed)?|live|radio edit|bonus track|deluxe).*", "", clean_title, flags=re.IGNORECASE)
    clean_title = re.sub(r"\(.*?remaster.*?\)", "", clean_title, flags=re.IGNORECASE)
    return primary_artist.strip(), clean_title.strip()

@app.get("/api/preview", tags=["Audio Preview"], summary="Fetch 30s audio preview stream URL")
def get_song_preview(
    track_name: str = Query(..., min_length=1, description="Track title"),
    artist: str = Query(..., min_length=1, description="Artist name"),
):
    """
    Returns a streaming 30s audio preview URL for a given track.
    Prioritizes iTunes (permanent, non-expiring AAC stream URLs).
    Falls back to Deezer (15-min tokenized MP3 stream URLs with automatic cache TTL).
    """
    cache_key = f"{artist.lower().strip()} - {track_name.lower().strip()}"
    now = time.time()
    if cache_key in preview_cache:
        entry = preview_cache[cache_key]
        if entry.get("expires_at", float("inf")) > now:
            return entry["data"]

    primary_artist, clean_title = clean_music_query(track_name, artist)
    queries = [
        f"{primary_artist} {clean_title}",
        f"{artist} {track_name}",
    ]
    # Deduplicate while preserving order
    queries = list(dict.fromkeys(q for q in queries if q.strip()))

    # 1. Primary: iTunes Search API (Permanent URLs with no token expiration)
    for q in queries:
        try:
            url = f"https://itunes.apple.com/search?term={urllib.parse.quote(q)}&entity=song&limit=1"
            resp = http_session.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results and results[0].get("previewUrl"):
                    res = {"status": "available", "preview_url": results[0]["previewUrl"], "source": "itunes"}
                    preview_cache[cache_key] = {"data": res, "expires_at": float("inf")}
                    return res
        except Exception:
            pass

    # 2. Fallback: Deezer API (Tokens expire in 15 mins; cache for 10 mins so links are always fresh)
    for q in queries:
        try:
            url = f"https://api.deezer.com/search?q={urllib.parse.quote(q)}&limit=1"
            resp = http_session.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                if results and results[0].get("preview"):
                    res = {"status": "available", "preview_url": results[0]["preview"], "source": "deezer"}
                    preview_cache[cache_key] = {"data": res, "expires_at": now + 600}
                    return res
        except Exception:
            pass

    res = {"status": "unavailable", "preview_url": None, "source": None}
    preview_cache[cache_key] = {"data": res, "expires_at": now + 3600}
    return res