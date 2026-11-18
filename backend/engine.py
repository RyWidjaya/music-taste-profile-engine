import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def taste_profile(track_ids, data):
    """
    Computes a comprehensive listener taste diagnostic from a list of Spotify track IDs.

    Calculates:
      1. Persona Breakdown: % distribution across the 7 Sound Personas.
      2. Diversity Score: Normalized Shannon entropy (0.0 = focused, 1.0 = eclectic).
      3. Percentiles: Benchmark user's audio traits against the global population.
      4. Recommendations: Top 3 nearest-neighbor songs in the user's dominant persona.
      5. User Songs: Coordinates (pca_x, pca_y) for 2D Sound Map visualization.

    Parameters:
      track_ids (list[str]): List of unique Spotify track IDs.
      data (pd.DataFrame): Modeled dataset loaded in RAM with features, clusters & PCA coordinates.

    Returns:
      dict: Structured taste diagnostic matching the TasteProfileResponse schema.
    """
    
    selected = data[data['track_id'].isin(track_ids)]

    # Persona Breakdown - % Breakdown of each Cluster 
    breakdown = selected['persona'].value_counts(normalize=True).to_dict()

    # Diversity Score - Shannon Entropy
    diversity_score = 0
    for i in breakdown: 
        diversity_score += (-breakdown[i]) * np.log(breakdown[i]) # calculate diversity score by summing for all clusters

    diversity_score /= np.log(7) # divide by the maximum entropy which is log(7) - perfectly split taste amongst all 7 clusters

   # Percentiles 
    percentiles = {}
    compare_features = ['danceability', 'energy', 'valence', 'acousticness'] # main features we are interested in for statistics

    for feat in compare_features:
        user_avg = selected[feat].mean()
        # calculate percentile of a value (user_avg) compared to list of values (data[feat])
        percentiles[feat] = round(float(percentileofscore(data[feat], user_avg)), 1)

    # Recommendations - Recommend 3 songs from the user's most common cluster (cluster with most songs that belong to it from the user)
    top_cluster = selected['cluster'].mode()[0] # most popular cluster 
    top_cluster_features = selected[selected['cluster'] == top_cluster] # select only from the top cluster
    mean_point = (top_cluster_features['pca_x'].mean(), top_cluster_features['pca_y'].mean()) # calculate mean point of all the user's points within that cluster

    other_songs = data[(data['cluster'] == top_cluster) & (~data['track_id'].isin(track_ids))].copy() # select every other song to create a new df and assign to a new variable

    other_songs['dist'] = (other_songs['pca_x'] - mean_point[0])**2 + (other_songs['pca_y'] - mean_point[1])**2 # calculate euclidean distance of all points to the mean point

    cols = ['track_id', 'track_name', 'artists', 'album_name'] # these are the features we want to return in the recommendation
    # return the top 3 songs within that cluster (smallest euclidean distances), select only required features and return as a dict
    suggested_songs = other_songs.sort_values(by='dist', ascending=True)[cols].head(3).to_dict(orient='records') # orient='records' treats every row as a dict (record)

    # 2D Sound Map Coordinates for the user's selected songs
    user_songs = selected[['track_id', 'track_name', 'artists', 'pca_x', 'pca_y', 'persona']].to_dict(orient='records')

    # return a dict of all dicts/stats
    return {
        "persona_breakdown": breakdown,
        "diversity_score": round(float(diversity_score), 2),
        "percentiles": percentiles,
        "recommendations": suggested_songs,
        "user_songs": user_songs
    }