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

    other_songs['dist'] = np.sqrt((other_songs['pca_x'] - mean_point[0])**2 + (other_songs['pca_y'] - mean_point[1])**2) # calculate euclidean distance of all points to the mean point

    other_songs['tempo'] = other_songs['tempo'].round().astype(int)
    other_songs['energy_pct'] = (other_songs['energy'] * 100).round().astype(int)
    other_songs['valence_pct'] = (other_songs['valence'] * 100).round().astype(int)
    other_songs['acoustic_pct'] = (other_songs['acousticness'] * 100).round().astype(int)
    other_songs['dance_pct'] = (other_songs['danceability'] * 100).round().astype(int)

    # Sort to get the top 3 closest recommendations
    top_3_recs = other_songs.sort_values(by='dist', ascending=True).head(3).copy()
    d0 = top_3_recs.iloc[0]['dist'] if len(top_3_recs) > 0 else 0
    top_3_recs['match_pct'] = [
        int(np.clip(round(99 - (rank * 2.5) - ((row['dist'] - d0) * 100)), 85, 99))
        for rank, (_, row) in enumerate(top_3_recs.iterrows())
    ]

    cols = [
        'track_id', 'track_name', 'artists', 'album_name',
        'tempo', 'energy_pct', 'valence_pct', 'acoustic_pct', 'dance_pct',
        'track_genre', 'match_pct'
    ] # these are the features we want to return in the recommendation
    suggested_songs = top_3_recs[cols].to_dict(orient='records')

    # 2D Sound Map Coordinates for the user's selected songs
    user_songs_cols = ['track_id', 'track_name', 'artists', 'pca_x', 'pca_y', 'persona']
    if 'album_name' in selected.columns:
        user_songs_cols.append('album_name')
    user_songs = selected[user_songs_cols].to_dict(orient='records')

    # Anchor & Outlier Track Identification
    # - Anchor: The track with the smallest combined Euclidean distance to all your other selected songs
    # - Outlier: The track with the largest combined Euclidean distance from all your other selected songs
    coords = selected[['pca_x', 'pca_y']].to_numpy()
    n_tracks = len(coords)
    anchor_track = None
    outlier_track = None

    def _track_dict(row):
        return {
            'track_id': str(row['track_id']),
            'track_name': str(row['track_name']),
            'artists': str(row['artists']),
            'album_name': str(row.get('album_name', '')),
            'persona': str(row['persona']),
            'pca_x': float(row['pca_x']),
            'pca_y': float(row['pca_y']),
        }

    if n_tracks > 0:
        dominant_persona = str(selected['persona'].mode()[0])

        if n_tracks == 1:
            anchor_track = _track_dict(selected.iloc[0])
            outlier_track = None
        elif n_tracks == 2:
            # When n < 3: Two songs are equidistant, so neither is an outlier.
            # Hide the Outlier completely, displaying only Anchor.
            # Tie-break: prefer the track belonging to the user's dominant cluster.
            p0_dom = str(selected.iloc[0]['persona']) == dominant_persona
            p1_dom = str(selected.iloc[1]['persona']) == dominant_persona
            anchor_idx = 1 if (not p0_dom and p1_dom) else 0
            anchor_track = _track_dict(selected.iloc[anchor_idx])
            outlier_track = None
        else:
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
            combined_dists = dist_matrix.sum(axis=1)

            # Anchor: smallest combined Euclidean distance
            # Tie-break: prefer the track belonging to the user's dominant cluster
            min_dist = np.min(combined_dists)
            min_candidates = np.where(np.isclose(combined_dists, min_dist, atol=1e-6))[0]
            if len(min_candidates) > 1:
                dom_candidates = [idx for idx in min_candidates if str(selected.iloc[idx]['persona']) == dominant_persona]
                min_idx = dom_candidates[0] if dom_candidates else int(min_candidates[0])
            else:
                min_idx = int(min_candidates[0])

            # Outlier: largest combined Euclidean distance, strictly enforcing outlierIndex !== anchorIndex
            dists_for_outlier = combined_dists.copy()
            dists_for_outlier[min_idx] = -np.inf
            max_dist = np.max(dists_for_outlier)
            max_candidates = [idx for idx in range(n_tracks) if idx != min_idx and np.isclose(dists_for_outlier[idx], max_dist, atol=1e-6)]
            if len(max_candidates) > 1:
                non_dom = [idx for idx in max_candidates if str(selected.iloc[idx]['persona']) != dominant_persona]
                max_idx = non_dom[0] if non_dom else int(max_candidates[0])
            else:
                max_idx = int(max_candidates[0])

            anchor_track = _track_dict(selected.iloc[min_idx])
            outlier_track = _track_dict(selected.iloc[max_idx])

    # return a dict of all dicts/stats
    return {
        "persona_breakdown": breakdown,
        "diversity_score": round(float(diversity_score), 2),
        "percentiles": percentiles,
        "recommendations": suggested_songs,
        "user_songs": user_songs,
        "anchor": anchor_track,
        "outlier": outlier_track,
    }