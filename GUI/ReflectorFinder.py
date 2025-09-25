from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime


def findClusters(points):
    """Use DBSCAN to cluster points and identify reflectors"""
    if len(points) == 0:
        return np.array([])
    
    dbscan = DBSCAN(eps=10, min_samples=15)
    cluster_flags = dbscan.fit_predict(points)

    return cluster_flags


def findReflectors(points, cluster_flags, max_radius = 120):
    """Extract reflector positions from clusters"""
    reflectors = []
    for cluster_id in set(cluster_flags):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_points = points[cluster_flags == cluster_id]
        centroid = cluster_points.mean(axis=0)

        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        if dists.max() > max_radius:
            continue  # reject large / long linear cluster instead of punctual

        reflectors.append((cluster_id, centroid))
    return reflectors


def calcConfidence(cluster_events, max_freq=25, max_lgv=10):
    """Calculate confidence score for each reflector based on events"""
    
    if not cluster_events:
        return 0.0
    
    # 1. Frequency score
    steepness = 0.8  # steepness of the sigmoid
    tolerance = 0.99  # tolerance to reach max score
    shift = max_freq - (np.log((1 / tolerance) - 1) / steepness)  # shift to reach max score at max_freq
    count = len(cluster_events)
    freq_score = 1 / (1 + np.exp(-steepness * (count - shift)))
    
    # 2. LGV Diversity Score (multiple vehicles = higher significance)
    lgvs = [e[0] for e in cluster_events]
    unique_lgvs = len(set(lgvs))
    lgv_score = min(1.0, unique_lgvs / max_lgv)  # Peak score when events recorded by max_lgv LGVs
    
    # 3. Time Concentration Score (events closer in time = lower score)
    timestamps = [datetime.fromisoformat(e[1]) for e in cluster_events]
    if len(timestamps) > 1:
        seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]
        time_span = max(seconds) - min(seconds) + 1  # Avoid division by zero
        # Inverse relationship: smaller time span = higher score
        time_score = min(1.0, time_span / 3600)  # Peak score when events distanced by >60 minutes
    else:
        time_score = 1.0
    
    # Combine scores with weights
    confidence_score = (0.20 * freq_score) + (0.5 * lgv_score) + (0.30 * time_score)
    return confidence_score


def analyzeReflectors(points_array, all_events, confidence_threshold=0.3):
    """
    Main workflow function to analyze reflectors from point data
    
    Args:
        points_array: numpy array of (x, y) coordinates
        all_events: list of events with full data (lgv_id, timestamp, worldx, worldy, lgvx, lgvy)
        confidence_threshold: minimum confidence score to include reflector
    
    Returns:
        list of tuples: (cluster_id, centroid, confidence_score)
    """
    if len(points_array) == 0:
        print("[WARNING] No points provided for analysis")
        return []
    
    print(f"[INFO] Analyzing {len(points_array)} points for reflectors...")
    
    # Find clusters
    cluster_flags = findClusters(points_array)
    
    if len(cluster_flags) == 0:
        print("[WARNING] No clusters found")
        return []
    
    # Count clusters (excluding noise)
    unique_clusters = set(cluster_flags)
    noise_count = np.sum(cluster_flags == -1)
    cluster_count = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    print(f"[INFO] Found {cluster_count} clusters, {noise_count} noise points")
    
    # Extract reflectors
    reflectors = findReflectors(points_array, cluster_flags)
    
    if not reflectors:
        print("[WARNING] No valid reflectors found")
        return []
    
    # Calculate confidence for each reflector
    reflector_scores = []
    for cluster_id, centroid in reflectors:
        # Get events belonging to this cluster
        cluster_events = [all_events[i] for i, flag in enumerate(cluster_flags) if flag == cluster_id]
        
        # Calculate confidence
        confidence = calcConfidence(cluster_events)
        
        # Only include reflectors above threshold
        if confidence >= confidence_threshold:
            reflector_scores.append((cluster_id, centroid, confidence))
            print(f"[INFO] Reflector {cluster_id}: centroid=({centroid[0]:.1f}, {centroid[1]:.1f}), confidence={confidence:.3f}")
        else:
            print(f"[DEBUG] Reflector {cluster_id}: confidence={confidence:.3f} below threshold {confidence_threshold}")
    
    print(f"[INFO] Found {len(reflector_scores)} high-confidence reflectors")
    return reflector_scores


def layoutCompare(layoutReflectors, clusters):
    """Compare the found clusters with the reflectors in the layout"""
    # TO BE DEVELOPED LATER
    pass