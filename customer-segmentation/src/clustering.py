import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]

def prepare_features(rfm):
    if rfm.empty:
        raise ValueError("RFM table is empty.")
    x = rfm[FEATURES].astype(float).copy()
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("RFM features contain non-finite values.")
    # RFM is usually right-skewed; log1p limits the influence of large values.
    x = np.log1p(x.clip(lower=0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    return scaled, scaler, x

def evaluate_cluster_counts(scaled, min_k=2, max_k=8, random_state=42):
    n = len(scaled)
    if n < 3:
        raise ValueError("At least 3 customers are required to evaluate clustering.")
    upper = min(max_k, n - 1)
    results = []
    for k in range(min_k, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        results.append({"k": k, "inertia": float(model.inertia_),
                        "silhouette": float(silhouette_score(scaled, labels))})
    if not results:
        raise ValueError("No valid cluster-count evaluations could be computed.")
    return pd.DataFrame(results)

def fit_clusters(rfm, k, random_state=42):
    if len(rfm) <= k:
        raise ValueError("Customer count must be greater than cluster count.")
    scaled, scaler, transformed = prepare_features(rfm)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled)
    if len(set(labels)) < 2:
        raise ValueError("K-Means produced fewer than two distinct clusters.")
    out = rfm.copy()
    out["Cluster"] = labels
    return out, model, scaler, transformed
