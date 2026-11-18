# Full-Stack Data Flow & Tech Stack Architecture Guide

A comprehensive, beginner-friendly reference detailing **every technology in our stack** and tracing **the exact step-by-step path of data** as it travels between the browser, the web server, and the machine learning model.

---

## 1. High-Level System Map

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                 TIER 3: FRONTEND                                  │
│                       (HTML5 / TailwindCSS / JavaScript)                          │
│                                                                                   │
│   [ Search Bar ]             [ Donut Chart ]             [ Spotify Players ]      │
│   (User types "queen")       (Persona percentages)       (Playable recommendations)│
└────────────┬──────────────────────────────────────────────────────▲───────────────┘
             │                                                      │
             │ HTTP Request                                         │ HTTP Response
             │ (JSON / Query params)                                │ (Structured JSON)
             ▼                                                      │
┌───────────────────────────────────────────────────────────────────┴───────────────┐
│                                 TIER 2: BACKEND                                   │
│                           (FastAPI / Uvicorn / Pydantic)                          │
│                                                                                   │
│   ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐   │
│   │   Uvicorn (Server)  │───►│  Pydantic (Schemas) │───►│   FastAPI (Routes)  │   │
│   │   Listens on :8000  │    │  Validates data types│   │   Executes logic    │   │
│   └─────────────────────┘    └─────────────────────┘    └──────────┬──────────┘   │
└────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                     │
                                     Direct Memory Access (RAM)      │ Query & Infer
                                     (< 1 millisecond latency)       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                  TIER 1: ML ENGINE                                │
│                       (Pandas / Scikit-Learn / Joblib in RAM)                     │
│                                                                                   │
│   ┌───────────────────────────────┐       ┌───────────────────────────────────┐   │
│   │   Modeled DataFrame (89.5k)   │       │     Frozen Joblib Brains          │   │
│   │   (Audio features, clusters,  │       │  - KMeans(k=7)                    │   │
│   │    personas, 2D coordinates)  │       │  - PowerTransformer, PCA models   │   │
│   └───────────────────────────────┘       └───────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack Encyclopedia

Every tool in our project was chosen for a specific architectural reason. Here is what each one does and why it was selected:

### Tier 1: Machine Learning & Statistical Engine
| Technology | Role in this Project | Why We Chose It |
| :--- | :--- | :--- |
| **Python (3.10+)** | Core programming language | The undisputed lingua franca of data science, ML, and AI. |
| **Pandas** | In-memory data manipulation | Provides vectorized filtering and indexing across 89,583 songs at C-speed. |
| **NumPy** | Vector math & array operations | Powers distance formulas, logarithmic entropy math, and coordinate arrays. |
| **Scikit-Learn** | ML modeling & transformations | Industry-standard implementations of $k$-Means clustering, PCA, and PowerTransformer. |
| **SciPy (`stats`)** | Population percentiles | `percentileofscore` benchmarks a user's sound traits against the global distribution. |
| **Joblib** | Model serialization | Efficiently freezes trained Scikit-learn estimator weights into disk files (`.joblib`). |

### Tier 2: Backend & Model Serving
| Technology | Role in this Project | Why We Chose It |
| :--- | :--- | :--- |
| **FastAPI** | REST API framework | Built for high-speed Python serving; auto-generates documentation; async-first. |
| **Uvicorn** | ASGI web server | The lightning-fast engine that runs FastAPI and listens for incoming network connections. |
| **Pydantic (`BaseModel`)** | Data validation & contracts | Guarantees incoming and outgoing JSON adheres strictly to expected types. |
| **CORS Middleware** | Security handshake | Grants permission for browser frontends on other ports to talk to our API. |

### Tier 3: Frontend & Visualization
| Technology | Role in this Project | Why We Chose It |
| :--- | :--- | :--- |
| **HTML5** | Semantic structure | Defines the skeleton of the webpage (search inputs, chart canvases, cards). |
| **TailwindCSS** | Utility-first styling | Rapidly styles dark-mode modern UIs using clean CSS class names without writing raw CSS. |
| **Vanilla JavaScript (ES6)** | Browser logic & API client | Uses the native `fetch()` API to send HTTP requests and update the DOM dynamically. |
| **Chart.js** | Data visualization | Draws interactive, animated donut charts and 2D scatter plots in HTML5 `<canvas>`. |
| **Spotify IFrame Embeds** | Multimedia playback | Renders official, playable Spotify audio preview widgets without needing Spotify API keys. |

---

## 3. Deep Dive: Data Flow 1 — Autocomplete Song Search

Here is what happens under the hood when a user types a song name into the search bar:

```
[ User types "Daft" in browser ]
       │
       ▼
1. JavaScript Event Listener captures keystroke
       │
       ▼
2. JS fetch() sends HTTP GET request:
   GET http://127.0.0.1:8000/api/search?q=Daft&limit=10
       │
       ▼
3. Uvicorn receives TCP network packets on Port 8000
       │
       ▼
4. FastAPI matches route: @app.get("/api/search")
   - Extracts query parameter: q = "Daft"
   - Validates query constraints (min_length=1)
       │
       ▼
5. In-Memory Pandas Vectorized Search:
   mask = data["track_name"].str.contains("daft", case=False) | ...
   matches = data[mask].sort_values(by="popularity", ascending=False).head(10)
       │
       ▼
6. Serialization:
   matches.to_dict(orient="records") transforms rows into Python dicts
       │
       ▼
7. Pydantic validates output against TrackSearchResult schema
       │
       ▼
8. Uvicorn sends HTTP 200 OK response with JSON payload:
   [
     { "track_id": "0xF9...", "track_name": "One More Time", "artists": "Daft Punk", ... },
     ...
   ]
       │
       ▼
9. Browser receives JSON -> JS loops over results -> Paints dropdown list in DOM
```

### Key Intuition: Why is this search so fast?
Notice that step 5 **does not touch your hard drive**. Because `artifacts["data"]` was loaded into your computer's RAM during server boot, searching 89,583 songs takes **under 4 milliseconds**.

---

## 4. Deep Dive: Data Flow 2 — The Taste Profile Diagnostic

Here is what happens when a user selects 5 songs and clicks **"Analyze My Sound"**:

```
[ User clicks "Analyze My Sound" ]
       │
       ▼
1. JavaScript gathers selected track IDs:
   const payload = {
     track_ids: ["5SuOikwiRyPMVoIQDJUgSV", "4UkSQn96mP2pGjK0iFq3wL", ...]
   };
       │
       ▼
2. JS calls fetch() with HTTP POST:
   fetch("http://127.0.0.1:8000/api/taste-profile", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify(payload)
   })
       │
       ▼
3. FastAPI & Pydantic intercept request:
   - Verifies track_ids is a non-empty list of strings (TasteProfileRequest)
   - If invalid -> instantly halts and returns HTTP 422
       │
       ▼
4. Pandas in-memory filter:
   selected = data[data["track_id"].isin(request.track_ids)]
       │
       ▼
5. The ML Mathematical Diagnostics Execute:
   ├─► A. Persona Breakdown:
   │      selected["persona"].value_counts(normalize=True)
   │
   ├─► B. Shannon Entropy Diversity Score:
   │      H = -sum(p * ln(p)) / ln(7)
   │
   ├─► C. Population Percentiles:
   │      percentileofscore(data[feature], user_avg) for each audio trait
   │
   ├─► D. Within-Cluster Recommendations:
   │      Find top cluster -> compute user mean (pca_x, pca_y) ->
   │      Find 3 closest non-selected songs by Euclidean distance
   │
   └─► E. Sound Map Coordinates:
          Extract (pca_x, pca_y) for each user track
       │
       ▼
6. Dict packaged into TasteProfileResponse schema
       │
       ▼
7. Uvicorn transmits HTTP 200 OK with complete diagnostic JSON
       │
       ▼
8. Browser receives JSON:
   ├─► Chart.js renders the animated Persona Donut Chart
   ├─► Entropy Gauge displays diversity score (e.g., "0.82 - Eclectic")
   ├─► Percentile Bars animate (e.g., "Energy: Higher than 78% of songs")
   ├─► 2D Scatter Canvas plots user songs as glowing dots on the Sound Map
   └─► 3 Spotify <iframe> players load playable song cards
```

---

## 5. Core Concepts Explained for Beginners

### 1. What is Serialization vs. Deserialization?
* **Serialization (Object $\rightarrow$ Text)**: Converting an in-memory programming object (like a Python dict or Pandas row) into a portable text format (like a JSON string or binary `.joblib` file) that can be saved to disk or sent across the network.
* **Deserialization (Text $\rightarrow$ Object)**: Reading that text or binary stream back and reconstructing the live object in memory.

| Direction | What Happens | Example |
| :--- | :--- | :--- |
| **Model Saving** | Python KMeans object $\rightarrow$ `.joblib` file on disk | `joblib.dump(kmeans, 'models/kmeans_model.joblib')` |
| **Model Loading** | `.joblib` file on disk $\rightarrow$ Python KMeans object in RAM | `joblib.load('models/kmeans_model.joblib')` |
| **API Response** | Python dict $\rightarrow$ JSON text string | Handled automatically by FastAPI / Pydantic |
| **API Request** | JSON text string $\rightarrow$ Python dict / Pydantic object | Handled automatically by FastAPI / Pydantic |

---

### 2. Statelessness: Why Web Servers Don't "Remember" Users
In a notebook, variables persist forever in your session. 

In production, **APIs are stateless**:
* User A sends a request $\rightarrow$ Server answers $\rightarrow$ Server immediately forgets User A existed.
* User B sends a request 1 second later $\rightarrow$ Server answers $\rightarrow$ Server forgets User B.

This is a **feature, not a bug**. Because the server doesn't store session baggage in memory, 1 single FastAPI instance can handle thousands of different users without running out of RAM. Everything the server needs to know must arrive inside the incoming request payload (`track_ids`).

---

### 3. The In-Memory Architecture (RAM vs. Disk)
* **The Disk (Hard Drive)**: Permanent storage (`data/spotify_tracks_modeled.csv`, `models/*.joblib`). Reading from disk is slow (100–500 milliseconds).
* **The RAM (Memory)**: High-speed temporary workspace. Reading from RAM takes **nanoseconds**.

By using FastAPI's `@asynccontextmanager lifespan`:
1. When you type `uvicorn backend.main:app`, the server reads the files from disk **once** and loads them into RAM (`artifacts["data"]`).
2. While the server is running, **it never reads from disk again**. All user searches, calculations, and recommendations happen entirely inside RAM at lightning speed.

---

### 4. CORS (Cross-Origin Resource Sharing)
By default, web browsers have a security rule: a website loaded from `http://localhost:5173` (your frontend) is blocked from fetching data from `http://127.0.0.1:8000` (your backend) because the origins (ports) don't match.

Adding `CORSMiddleware` in `main.py` attaches an HTTP header to every response:
```http
Access-Control-Allow-Origin: *
```
This tells the user's browser: *"Hey browser! I explicitly permit this website to receive my JSON data."*

---

## 6. How to Run the Backend

To boot the backend API server locally, run:

```bash
uvicorn backend.main:app --reload
```

* **`backend.main`**: Tells Python to find the file `backend/main.py`.
* **`:app`**: Refers to the `app = FastAPI(...)` object inside that file.
* **`--reload`**: Developer mode! Automatically reboots the server whenever you edit and save code.

Once running, navigate to:
* **Interactive Testing UI (Swagger)**: `http://127.0.0.1:8000/docs`
* **Alternative Documentation (ReDoc)**: `http://127.0.0.1:8000/redoc`
* **Health Check Endpoint**: `http://127.0.0.1:8000/api/health`
