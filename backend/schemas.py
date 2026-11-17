from pydantic import BaseModel, Field
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# 1. Search Request & Response Schemas
# ---------------------------------------------------------------------------
class TrackSearchResult(BaseModel):
    track_id: str
    track_name: str
    artists: str
    album_name: str
    track_genre: str
    persona: str


# ---------------------------------------------------------------------------
# 2. Taste Profile Request Schema
# ---------------------------------------------------------------------------
class TasteProfileRequest(BaseModel):
    """
    The incoming payload from the frontend.
    The user sends a list of Spotify track_ids they selected from autocomplete.
    """
    track_ids: List[str] = Field(
        ..., 
        min_length=1, 
        description="List of Spotify track IDs to analyze."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "track_ids": [
                    "5SuOikwiRyPMVoIQDJUgSV",
                    "4UkSQn96mP2pGjK0iFq3wL",
                    "0VjIjW4GlUZAMYd2vXMi3b"
                ]
            }
        }
    }


# ---------------------------------------------------------------------------
# 3. Taste Profile Response Schema
# ---------------------------------------------------------------------------
class TasteProfileResponse(BaseModel):
    """
    The exact JSON contract returned to the frontend.
    Directly mirrors the dictionary produced by our ML taste_profile() engine.
    """
    persona_breakdown: Dict[str, float] = Field(
        ..., 
        description="Percentage of user songs belonging to each of the 7 personas."
    )
    diversity_score: float = Field(
        ..., 
        description="Normalized Shannon entropy (0.0 to 1.0) measuring taste diversity."
    )
    percentiles: Dict[str, float] = Field(
        ..., 
        description="User's average audio traits benchmarked against the 89k dataset."
    )
    recommendations: List[Dict[str, Any]] = Field(
        ..., 
        description="Top 3 nearest-neighbor songs within the user's dominant persona."
    )
    user_songs: List[Dict[str, Any]] = Field(
        ..., 
        description="User's songs with (pca_x, pca_y) coordinates for 2D Sound Map."
    )
