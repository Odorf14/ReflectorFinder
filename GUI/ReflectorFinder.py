from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime


def correctPointPos(events, reflector_radius = 32):
    """Offset point positions based on LGV heading and reflector radius"""
    """Asuming that the observation is a real reflector!!!"""
    corrected_points = []

    for event in events:
        lgv_id, timestamp, wx, wy, lgvx, lgvy = event
        reflector = np.array([wx, wy])
        lgv_pos = np.array([lgvx, lgvy])

        heading = reflector - lgv_pos # Vector from LGV to reflector
        unit_vector = heading / np.linalg.norm(heading)

        corrected_pos = reflector + unit_vector * reflector_radius
        corrected_points.append(corrected_pos)
    
    return np.array(corrected_points)

def findClusters(points):
    """Use DBSCAN to cluster points and identify reflectors"""
    if len(points) == 0:
        return np.array([])
    
    dbscan = DBSCAN(eps=3.5, min_samples=20)
    cluster_flags = dbscan.fit_predict(points)

    return cluster_flags


def findReflectors(points, cluster_flags, radius_filter = True, max_radius = 130, circular_filter = False, circularity = 0.15):
    """Extract reflector positions from clusters"""
    reflectors = []
    for cluster_id in set(cluster_flags):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_points = points[cluster_flags == cluster_id]
        
        if len(cluster_points) == 0:
            continue

        mean_centroid = cluster_points.mean(axis=0)
        median_centroid = np.median(cluster_points, axis=0)

        centroid_diff = np.linalg.norm(mean_centroid - median_centroid)
        scale = max(1e-6, np.linalg.norm(np.std(cluster_points, axis=0)))
        if centroid_diff / scale > 0.5:
            centroid = median_centroid
        else:
            centroid = mean_centroid

        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        if radius_filter:
            if (dists > max_radius).mean() > 0.1:
                continue  # reject large / long linear cluster instead of punctual
        
        if circular_filter:
            if dists.std() / dists.mean() > circularity:
                continue  # reject non-circular clusters            

        reflectors.append((cluster_id, centroid))
    return reflectors

def merge_nearby_reflectors(reflectors, merge_distance=100):
    """Merge reflectors that are too close together"""
    merged = []
    for cluster_id, centroid in reflectors:
        # Check if too close to existing reflectors
        too_close = False
        for _, existing_centroid in merged:
            if np.linalg.norm(centroid - existing_centroid) < merge_distance:
                too_close = True
                break
        if not too_close:
            merged.append((cluster_id, centroid))
    return merged


def calcConfidence(cluster_events, max_freq=30, max_lgv=10):
    """Calculate confidence score for each reflector based on events"""
    
    if not cluster_events:
        return 0.0
    
    # 1. Frequency score
    #steepness = 0.8  # steepness of the sigmoid
    #tolerance = 0.99  # tolerance to reach max score
    #shift = max_freq - (np.log((1 / tolerance) - 1) / steepness)  # shift to reach max score at max_freq
    count = len(cluster_events)
    #freq_score = 1 / (1 + np.exp(-steepness * (count - shift)))
    freq_score = min(1.0, count / max_freq) # linear instead of sigmoid
    
    # 2. LGV Diversity Score (multiple vehicles = higher significance)
    unique_lgvs = len(set(e[0] for e in cluster_events))
    lgv_score = min(1.0, unique_lgvs / max_lgv)  # Peak score when events recorded by max_lgv LGVs
    
    # 3. Time Concentration Score (events closer in time = lower score)
    timestamps = [datetime.fromisoformat(e[1]) for e in cluster_events]
    if len(timestamps) > 1:
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        # Penalize very short time spans (likely noise)
        time_score = min(1.0, time_span / 7200)  # 2 hours for max score
    else:
        time_score = 0.1  # Single events get low time score
    
    # 4. Add spatial consistency score
    positions = np.array([(e[2], e[3]) for e in cluster_events])
    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    spatial_score = 1.0 / (1.0 + distances.std() / 50.0)  # Lower if spread out


    # Combine scores with weights
    confidence = (0.2 * freq_score) + (0.35 * lgv_score) + (0.25 * time_score) + (0.2 * spatial_score)
    return confidence


def analyzeReflectors(points_array, all_events, confidence_threshold=0.3, offset_correction=False):
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
    
    if offset_correction:
        print("[INFO] Applying point position correction...")
        points_array = correctPointPos(all_events, reflector_radius=32)

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

    reflectors = merge_nearby_reflectors(reflectors, merge_distance=100)
    
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