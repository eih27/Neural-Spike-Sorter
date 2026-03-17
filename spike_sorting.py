"""
spike_sorting.py — Spike Sorting via PCA + K-Means
====================================================
Pipeline:
  1. Flatten multi-channel waveform snippets
  2. PCA dimensionality reduction (retain 95% variance)
  3. K-means clustering with automatic K selection (silhouette score)
  4. Cluster quality metrics: ISI violations, SNR, isolation distance
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SortingResult:
    """Output of the spike sorting pipeline."""
    labels: np.ndarray               # cluster label per spike
    pca_features: np.ndarray         # (n_spikes, n_components) PCA projection
    pca_model: PCA                   # fitted PCA model
    n_clusters: int                  # number of clusters found
    centroids: np.ndarray            # cluster centroids in PCA space
    quality: Dict[int, Dict]         # per-cluster quality metrics
    explained_variance: np.ndarray   # PCA explained variance ratios


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_features(
    waveforms: np.ndarray,
    n_components: int = 10,
    variance_threshold: float = 0.95,
) -> Tuple[np.ndarray, PCA]:
    """
    Flatten waveforms and project onto principal components.

    Parameters
    ----------
    waveforms : (n_spikes, n_channels, snippet_len)
    n_components : max components to consider
    variance_threshold : retain enough PCs to explain this fraction

    Returns
    -------
    features : (n_spikes, n_retained_components)
    pca : fitted PCA model
    """
    n_spikes = waveforms.shape[0]
    flat = waveforms.reshape(n_spikes, -1)

    # Standardize
    flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-12)

    pca = PCA(n_components=min(n_components, flat.shape[1]))
    features = pca.fit_transform(flat)

    # Keep only enough components for variance threshold
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_keep = max(2, min(n_keep, features.shape[1]))  # at least 2 for vis

    return features[:, :n_keep], pca


# ─── Clustering ──────────────────────────────────────────────────────────────

def cluster_spikes(
    features: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
    seed: int = 42,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    K-means clustering with automatic K selection via silhouette score.

    Returns
    -------
    labels, best_k, centroids
    """
    best_score = -1
    best_k = k_range[0]
    best_labels = None
    best_centroids = None

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(features, labels, sample_size=min(5000, len(labels)))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centroids = km.cluster_centers_

    return best_labels, best_k, best_centroids


# ─── Quality Metrics ─────────────────────────────────────────────────────────

def compute_quality_metrics(
    labels: np.ndarray,
    spike_indices: np.ndarray,
    waveforms: np.ndarray,
    fs: int,
    isi_violation_ms: float = 1.5,
) -> Dict[int, Dict]:
    """
    Compute per-cluster quality metrics.

    Metrics
    -------
    - n_spikes: count
    - isi_violation_rate: fraction of ISI < refractory period
    - snr: signal-to-noise ratio (peak-to-peak mean / noise std)
    - firing_rate_hz: mean firing rate
    """
    unique_labels = np.unique(labels)
    duration_s = (spike_indices[-1] - spike_indices[0]) / fs if len(spike_indices) > 1 else 1.0
    isi_thresh = isi_violation_ms / 1000.0 * fs  # in samples

    metrics = {}
    for label in unique_labels:
        mask = labels == label
        cluster_indices = spike_indices[mask]
        cluster_waveforms = waveforms[mask]
        n = int(mask.sum())

        # ISI violations
        if n > 1:
            isis = np.diff(cluster_indices)
            isi_violations = float(np.sum(isis < isi_thresh)) / (n - 1)
        else:
            isi_violations = 0.0

        # SNR: mean waveform peak-to-peak / std of residuals
        mean_wf = cluster_waveforms.mean(axis=0)
        residuals = cluster_waveforms - mean_wf
        peak_to_peak = mean_wf.max() - mean_wf.min()
        noise_std = residuals.std() + 1e-12
        snr = peak_to_peak / noise_std

        metrics[int(label)] = {
            "n_spikes": n,
            "isi_violation_rate": round(isi_violations, 4),
            "snr": round(float(snr), 2),
            "firing_rate_hz": round(n / duration_s, 2),
        }

    return metrics


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def sort_spikes(
    waveforms: np.ndarray,
    spike_indices: np.ndarray,
    fs: int,
    n_pca_components: int = 10,
    k_range: Tuple[int, int] = (2, 8),
    seed: int = 42,
) -> SortingResult:
    """Run the complete sorting pipeline."""
    features, pca = extract_features(waveforms, n_components=n_pca_components)
    labels, n_clusters, centroids = cluster_spikes(features, k_range=k_range, seed=seed)
    quality = compute_quality_metrics(labels, spike_indices, waveforms, fs)

    return SortingResult(
        labels=labels,
        pca_features=features,
        pca_model=pca,
        n_clusters=n_clusters,
        centroids=centroids,
        quality=quality,
        explained_variance=pca.explained_variance_ratio_,
    )
