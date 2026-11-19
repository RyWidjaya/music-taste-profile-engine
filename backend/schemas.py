"""
Defines the structure and required data types (schemas) for each request and response exchanged
between the frontend and backend whenever the data has inherent structure.
"""

# Enforces strict type checking at runtime
# Basemodel is the parent class that all data schemas inherit from
# Field is a helper function used to attach extra validation rules and metadata 
from pydantic import BaseModel, Field

# Allow us to specify nested data structures 
# e.g. List[str] - list containing only strings
#      Dict[str, float] - a dict whose keys are strings and values are floats
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# General Rule: Write a schema class for each request and response.
# 
# A schema class is required if the data being passed hsa inherent structure
# - Structured data (JSON payloads, lists) has internal shape and requires a 
#   schema class to validate and enforce data types 
#   (e.g. list of track_ids passed into pipeline - list has structure -> schema)

# - Plain, primitive/scalar datatype without internal structure (str, int, float, bool) 
#   (e.g. a simple search keyword ?q=queen in the URL - a string)
#   do not need a schema class and are handled directly as function arguments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Search Request & Response Schemas
#    Defines the structure and data types of data passing through responses
#    Specifies the dictionary of details returned when the track_name is passed
#    - User writes name into the search bar 
#    - The server will respond with the matching tracks and their details/data
#    - The ml pipeline operates by track_id and uses it for the profile request
# ---------------------------------------------------------------------------
class TrackSearchResult(BaseModel): # Inherits data schemas from BaseModel
    # Defines the dtypes for all details returned in the dict 
    # Since a dict is returned by by the name search, a schema is required 
    # In Pydantic a class is treated as a dictionary
    track_id: str
    track_name: str
    artists: str
    album_name: str
    track_genre: str
    persona: str


# ---------------------------------------------------------------------------
# 2. Taste Profile Request Schema
#    Defines the dtypes for all the track_ids passed through the ml_pipeline 
#    function. A list of track_ids is passed to the func so a schema is needed 
#    - Defines the payload sent from the frontend when "Analyze" is clicked
#    - Shields the ML model by validating at least 1 valid track_id is provided
# ---------------------------------------------------------------------------
class TasteProfileRequest(BaseModel):
    """
    The incoming payload from the frontend.
    The user sends a list of Spotify track_ids they selected from autocomplete.
    """
    # track_ids is defined as a list of strings 
    # Field attaches extra validation rules, constraints and documentation 
    track_ids: List[str] = Field(
        ...,  # specifies the field is required - must pass a non-empty list
        min_length=1, # min length (at least 1 track)
        description="List of Spotify track IDs to analyze." # documentation 
    )

    # Model_config: Optional class-level configuration dictionary in Pydantic.
    # Used purely for documentation and testing convenience.
    model_config = {
        # json_schema_extra: Tells Pydantic to inject extra metadata into the generated API docs (OpenAPI/Swagger)
        "json_schema_extra": {
            # example: Pre-populates the interactive Swagger UI (/docs) with real, clickable sample IDs
            # so developers can test the endpoint with 1 click instead of manually typing track IDs.
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
#    Defines the dtypes for all the calculated results returned by the ml func
#    The func returns a dictionary listing the exact details -> schema needed
#    - Defines the structured diagnostic report returned to the frontend
#    - Guarantees exact keys and types 
# ---------------------------------------------------------------------------
class TasteProfileResponse(BaseModel):
    """
    The exact JSON contract returned to the frontend.
    Directly mirrors the dictionary produced by our ML taste_profile() engine.
    """
    # ... specifies these fields cannot be empty - field is required to be returned
    #     this means that all 5 complete metrics MUST be returned
    # description just used as documentation for each of the fields returned
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
        description="Percentile (0-100) of the user's average audio traits against all 127,900 individual tracks in the dataset."
    )
    recommendations: List[Dict[str, Any]] = Field(
        ..., 
        description="Top 3 nearest-neighbor songs within the user's dominant persona."
    )
    user_songs: List[Dict[str, Any]] = Field(
        ..., 
        description="User's songs with (pca_x, pca_y) coordinates for 2D Sound Map."
    )