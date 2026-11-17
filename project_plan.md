# What's My Sound? — A Music Taste Breakdown Built on k-Means

A portfolio project that clusters 127,900+ real Spotify tracks (spanning up through 2026, with curated 3rd–5th gen K-Pop and modern hits) into data-driven "sound personas," then tells a user what percentage of their listening falls into each one — like a music-taste personality quiz, powered by unsupervised learning instead of a canned genre label.

**Why this beats a generic clustering demo:** it doesn't predict the `track_genre` column that's already in the data (that's classification wearing a clustering costume). It lets k-means *invent its own* taste categories from raw audio features, independent of genre tags, and then measures a real person against them. That's a cleaner story for an interview: *"I didn't predict existing genre labels — I let the model define its own taste categories, then built a tool to score a listener against them."*

---

## How to work through this: three independent layers

This project splits cleanly into three layers, each with a different relationship to your existing skills. Build them in this order — each layer only needs to know the *output shape* of the layer below it, not how that layer works internally:

| Tier | Phases | What it is | How to approach it |
| --- | --- | --- | --- |
| **Tier 1 — ML pipeline** | Phases 1–6 | Cleaning, feature prep, clustering, evaluation, and the taste-breakdown logic — all plain Python/pandas/scikit-learn, no web code anywhere | **Figure it out yourself.** This is the part you already have the underlying skills for (scaling, fit/transform, k-means, silhouette) from your practice sets — this project just adds the judgment calls real, messy data forces on you. Attempt each step from your own reasoning first, *then* check it against this plan. |
| **Tier 2 — Backend/API** | Phase 7 | Wrapping your already-working `taste_profile()` function in a FastAPI endpoint that responds to HTTP requests | Genuinely new territory — learn by building and understanding, not just pasting. Closer to MLOps/model-serving than "web dev," and worth understanding *why* it's structured the way it is. |
| **Tier 3 — Frontend** | Phase 8 | The actual webpage UI | Vibecode freely. Not your career direction, not something you'll be evaluated on in ML engineering interviews — just read and understand what gets generated. |
| — | Phase 9 | Deployment | Backend deployment is closer to Tier 2; frontend deployment is closer to Tier 3. |

**Your Tier 1 checkpoint:** you'll know your ML pipeline is actually done — not just "the model is fit" — when you can call `taste_profile(["some", "track", "ids"])` directly in a notebook cell and get back a correct, sensible dict, with zero web code anywhere in sight. Don't open FastAPI docs until you've hit that checkpoint.

---

## Dataset & Ingestion Pipeline

- **Sources:** 
  1. [114K Spotify Tracks](https://github.com/sai-chaitanya-reddy/spotify-tracks-dataset) — CC0 / public domain baseline dataset collected via the Spotify Web API.
  2. `spotify-huge-audio-features.parquet` — 56.2M-row Spotify warehouse dataset used for catalog augmentation.
- **Data Evolution & Ingestion Workflow:**
  - `01_cleaning.ipynb` — Deduplicates the baseline 114k Spotify dataset down to `data/spotify_tracks_clean.csv` (89,583 unique tracks after eliminating ~24K cross-genre duplicate tags and nulls).
  - `02_eda.ipynb` — Explores audio feature distributions, zero-inflation in speechiness/instrumentalness, and multicollinearity in the energy/loudness/acousticness triad.
  - `03_data_augmentation.ipynb` — Incremental ingestion pipeline that scans the 56M-row Parquet warehouse (`spotify-huge-audio-features.parquet`) in memory-safe PyArrow batches, filters for popularity, enforces strict exact-artist matching for Gen 3–5 K-Pop, and resolves true genres across 31,000+ known artists. Integrates curated 2026 title tracks (*Hearts2Hearts – Lemon Tang*, *NMIXX – Heavy Serenade*, *BABYMONSTER – DESERVE / CHOOM*, *aespa – Lemonade*, *IVE – Blackhole*, etc.) to gracefully handle Spotify's late-2024 deprecation of public audio features. Outputs `data/spotify_master_2026.csv` (127,900 unique tracks with 0 nulls across 20 schema-conformed features).
  - `04_ml_pipeline.ipynb` — Executes production Yeo-Johnson power transformations, 1D PCA intensity extraction, 2D PCA sound-map projections, optimal $k=7$ KMeans training, cluster persona naming, and the complete `taste_profile()` inference engine.
- **Columns you'll actually cluster on:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` (9 numeric audio-character features). Ignore `popularity`, `duration_ms`, `key`, `mode`, `time_signature` for clustering — they're metadata, not "sound character."
- **Known quality note:** silhouette scores on this data land around 0.17–0.20 for k = 4–8. That's normal and expected for real audio data (unlike the clean synthetic blobs in your practice sets) — state it honestly in your write-up rather than cherry-picking a flattering number. It's also a good excuse to bring in the validation techniques in Phase 5.

---

## Phase 1 — Clean & explore *(Tier 1 — ML pipeline)*

Load `spotify_tracks_clean.csv`. Check distributions of the 9 audio features — several are skewed (`instrumentalness` and `speechiness` are heavily zero-inflated, `loudness` is left-skewed). Plot a correlation heatmap between features to check for redundancy.

## Phase 2 — Preprocess *(Tier 1 — ML pipeline)*

Log-transform or winsorize the heavy-tailed features. Fit a `StandardScaler` on the 9 chosen features — the exact skill from your earlier practice sets. Save the fitted scaler with `joblib`; you'll reuse it in the backend to transform any new song's feature vector.

## Phase 3 — Reduce dimensions for visualization *(Tier 1 — ML pipeline)*

Fit `PCA` to 2 components on the scaled features so the full dataset can be plotted as a 2D scatter. Report explained variance. This projection becomes the interactive "map of all sound" in your app.

## Phase 4 — Choose k and fit the model *(Tier 1 — ML pipeline)*

Run the elbow method (inertia vs. k) and a silhouette sweep (k = 3 to 12) like your practice problems, but on a subsample for speed since `silhouette_score` is O(n²). Pick k with a documented justification, not just "highest silhouette" — cross-check against how many genres are actually present. Fit the final `KMeans` on all 127,900 tracks and save it with `joblib`.

## Phase 5 — Name and validate the clusters *(Tier 1 — ML pipeline)*

**The problem:** `kmeans.labels_` gives you integers (0, 1, 2, 3) with no inherent meaning. Interpreting a cluster means figuring out, after the fact, what made the algorithm group those points together.

**Why raw cluster means aren't enough:** comparing a cluster's average `loudness` (say, -5.2) to its average `valence` (say, 0.6) is meaningless on its own — they're different units on different scales, the same problem that motivated scaling before clustering in the first place.

**The fix — z-score the cluster averages against the global distribution, the same formula as `StandardScaler`, just applied one level up:**

```python
cluster_profile = df.groupby('cluster')[feature_cols].mean()
global_mean = df[feature_cols].mean()
global_std = df[feature_cols].std()
cluster_zscores = (cluster_profile - global_mean) / global_std
```

A z-score of +2.0 on `acousticness` and -1.7 on `energy` are now directly comparable — both say "this many standard deviations from the typical track" — which raw means never let you do.

**Rank by absolute z-score, not just high values** — a cluster can be just as distinguished by being unusually *low* on something (e.g. near-zero `speechiness`) as by being unusually high:

```python
for c in cluster_zscores.index:
    top = cluster_zscores.loc[c].abs().sort_values(ascending=False).head(3)
    print(c, top)
```

The top 2-3 features by magnitude are what actually define that cluster relative to everything else — that's what the name should come from, not the full 9-feature profile.

**Corroborate, then spot-check:**

1. Cross-tabulate cluster membership against `track_genre` — if a cluster's top distinguishing features are high-acousticness/low-energy *and* its most common genre tags are acoustic/folk/singer-songwriter, those two signals agreeing is good evidence your read is right. Genre is corroborating evidence here, never the source of the name — the clustering itself never sees it.
2. Pull 5-10 real track names from the cluster and sanity-check by ear whether the name actually feels right, not just statistically justified.
3. Report the Adjusted Rand Index between your clusters and `track_genre` as a final external sanity check — framed explicitly as "genres are a noisy, human-labeled proxy, not ground truth," not as a claim your clusters should reproduce them exactly.

Expect at least one cluster to resist a clean name — that's a real signal, not a failure. Name it honestly as a catch-all (e.g. "Mixed/Eclectic") rather than forcing a tidier story than the data supports.

## Phase 6 — Build the taste-breakdown engine *(Tier 1 — ML pipeline)*

This is the feature that makes the project unique:

1. User supplies a list of songs (see "Getting songs from the user" below).
2. Look up each song's cluster with the already-fitted `.predict()` (never refit on the user's songs).
3. Aggregate into a **percentage breakdown** across personas — "38% High-Energy Pop, 27% Moody Acoustic, 20% Chill Instrumental, 15% Sad Bangers" — shown as a donut chart with a headline result, like a personality-quiz output.
4. **Compare to the global average.** Use `scipy.stats.percentileofscore` on each raw feature to say things like "your acousticness is higher than 78% of listeners in the dataset" — far more interesting than a bare percentage.
5. **Taste-diversity score.** Compute the Shannon entropy of the user's cluster-percentage distribution. High entropy = eclectic listener; low entropy = narrow, focused taste. A small, slightly unusual metric that signals you're thinking past "just run the model."
6. **Recommendations.** Using the within-cluster nearest-neighbor index (cosine similarity on scaled features), surface unheard songs from the user's dominant persona.

### Getting songs from the user — pick your complexity

- **Simple (recommended to start):** a search box where the user types/selects \~10–20 song names from the 128K dataset via autocomplete. No auth, no external API calls, no rate limits. Ships in a weekend on top of the pipeline you already have.
- **Stretch (do this later if time allows):** Spotify OAuth login to pull the user's actual playlist/top-tracks track IDs (this endpoint is still open), then join those IDs against your local dataset by `track_id` to fetch audio features — skip Spotify's `audio-features`/`audio-analysis` endpoints entirely, since Spotify locked those down for new developer apps in late 2024. Expect real coverage gaps (not every song a user picks will be in your 128K-track dataset) — state that honestly rather than hiding it.

## Phase 7 — Backend API *(Tier 2 — new territory, build + understand)*

Wrap the saved scaler + PCA + KMeans + within-cluster nearest-neighbor index in a small FastAPI (or Flask) service. Suggested endpoints: search tracks by name, get a track's persona, and a `/taste-profile` endpoint that accepts a list of track IDs and returns the full percentage breakdown, percentile comparisons, entropy score, and recommendations in one response. Load all models once at startup, not per-request.

## Phase 8 — Frontend *(Tier 3 — vibecode freely)*

A search box to build a list of songs → a "Your Sound" result screen with the donut-chart breakdown, headline persona, entropy/diversity score, percentile call-outs, and a list of recommended tracks. Add the PCA scatter of the full dataset colored by cluster, with the user's songs highlighted on it. Every row has a real Spotify `track_id`, so you can embed Spotify's official player with a plain iframe (`open.spotify.com/embed/track/{id}`) — no API key needed, and it makes the demo feel like a real product.

## Phase 9 — Deploy & document *(backend half = Tier 2, frontend half = Tier 3)*

Frontend on Vercel/Netlify, backend on Render/Railway/Fly.io — or bundle the whole thing as one Streamlit app on Streamlit Community Cloud for the fastest path to a live link. Write a README covering your methodology, the silhouette/ARI numbers, screenshots, and a live demo link. That README is most of what a recruiter actually reads.

---

## Stretch goals (only if you have time left)

- **Compare algorithms** — fit `GaussianMixture` or `AgglomerativeClustering` alongside `KMeans` and justify your final choice with metrics, not habit.
- **Stability check** — refit `KMeans` with several random seeds, measure Adjusted Rand Index between runs to show your clusters aren't an artifact of initialization.
- **Real Spotify login** — the OAuth stretch path described in Phase 6.

## Suggested tech stack

- **ML/data:** pandas, scikit-learn, joblib
- **Backend:** FastAPI (or Flask)
- **Frontend:** React or plain HTML/JS + a charting library (Chart.js/Recharts) — vibecode this layer
- **Deployment:** Vercel/Netlify (frontend) + Render/Railway/Fly.io (backend), or Streamlit Community Cloud as a single-app shortcut

## Draft resume bullet (tune once built)

*"Built and deployed an unsupervised music-taste analysis web app clustering 128K+ Spotify tracks by audio characteristics (k-means, PCA, scikit-learn); designed a taste-breakdown engine that scores a listener's song history against data-derived sound personas using percentile comparisons and Shannon-entropy diversity scoring, with a content-based recommender surfaced from within-cluster nearest neighbors."*