"""
spike_detection.py — Spike Detection Pipeline
==============================================
Two detection methods:
  1. Threshold-based (median absolute deviation)
  2. Wavelet-based (stationary wavelet transform)
Both include dead-time enforcement and waveform snippet extraction.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedSpikes:
    """Output of spike detection."""
    indices: np.ndarray          # sample indices of detected spike peaks
    waveforms: np.ndarray        # (n_spikes, n_channels, snippet_len)
    detection_channel: int       # channel used for detection
    threshold: float             # threshold value used


# ─── Bandpass Filter ─────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = 300.0,
    high: float = 6000.0,
    order: int = 3,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to all channels."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[0]):
        filtered[ch] = filtfilt(b, a, signal[ch])
    return filtered


# ─── Threshold Detection ────────────────────────────────────────────────────

def _mad_threshold(x: np.ndarray, n_sigma: float = 4.0) -> float:
    """Robust threshold using Median Absolute Deviation."""
    mad = np.median(np.abs(x - np.median(x)))
    sigma_est = mad / 0.6745
    return n_sigma * sigma_est


def detect_threshold(
    signal: np.ndarray,
    fs: int,
    n_sigma: float = 4.0,
    dead_time_ms: float = 1.0,
    snippet_before: int = 20,
    snippet_after: int = 40,
    detection_channel: Optional[int] = None,
) -> DetectedSpikes:
    """
    Threshold-based spike detection on a single channel.
    Detects negative-going crossings.
    """
    n_channels, n_samples = signal.shape

    # Auto-select channel with highest variance if not specified
    if detection_channel is None:
        detection_channel = int(np.argmax(np.var(signal, axis=1)))

    trace = signal[detection_channel]
    thresh = _mad_threshold(trace, n_sigma)

    # Find threshold crossings (negative peaks)
    below = trace < -thresh
    crossings = np.where(np.diff(below.astype(int)) == 1)[0]

    # Refine to local minima within a small window
    dead_samples = int(dead_time_ms / 1000.0 * fs)
    refined = []
    last = -dead_samples - 1
    for c in crossings:
        lo = max(0, c)
        hi = min(n_samples, c + snippet_after // 2)
        peak = lo + np.argmin(trace[lo:hi])
        if peak - last >= dead_samples:
            refined.append(peak)
            last = peak

    indices = np.array(refined, dtype=int)

    # Extract waveform snippets
    snippet_len = snippet_before + snippet_after
    valid = (indices >= snippet_before) & (indices < n_samples - snippet_after)
    indices = indices[valid]

    waveforms = np.zeros((len(indices), n_channels, snippet_len))
    for i, idx in enumerate(indices):
        waveforms[i] = signal[:, idx - snippet_before : idx + snippet_after]

    return DetectedSpikes(
        indices=indices,
        waveforms=waveforms,
        detection_channel=detection_channel,
        threshold=thresh,
    )


# ─── Wavelet Detection ──────────────────────────────────────────────────────

def detect_wavelet(
    signal: np.ndarray,
    fs: int,
    n_sigma: float = 3.5,
    wavelet: str = "db4",
    level: int = 4,
    dead_time_ms: float = 1.0,
    snippet_before: int = 20,
    snippet_after: int = 40,
    detection_channel: Optional[int] = None,
) -> DetectedSpikes:
    """
    Wavelet-based spike detection using Stationary Wavelet Transform.
    Uses detail coefficients at the scale matching spike frequency band.
    Falls back to threshold-based if pywt is unavailable.
    """
    try:
        import pywt
    except ImportError:
        print("[WARN] pywt not installed — falling back to threshold detection.")
        return detect_threshold(signal, fs, n_sigma, dead_time_ms,
                                snippet_before, snippet_after, detection_channel)

    n_channels, n_samples = signal.shape
    if detection_channel is None:
        detection_channel = int(np.argmax(np.var(signal, axis=1)))

    trace = signal[detection_channel]

    # Pad to power of 2 for SWT
    pad_len = int(2 ** np.ceil(np.log2(n_samples))) - n_samples
    padded = np.pad(trace, (0, pad_len), mode="reflect")

    # Stationary wavelet transform
    coeffs = pywt.swt(padded, wavelet, level=level, trim_approx=True)
    # Use detail coefficients at level that captures spike-band energy
    # Level 3-4 for 30kHz ≈ 937-1875 Hz — good spike range
    detail = coeffs[-min(level, 3)][:n_samples]

    # Threshold on wavelet detail
    thresh = _mad_threshold(detail, n_sigma)

    below = np.abs(detail) > thresh
    crossings = np.where(np.diff(below.astype(int)) == 1)[0]

    dead_samples = int(dead_time_ms / 1000.0 * fs)
    refined = []
    last = -dead_samples - 1
    for c in crossings:
        lo = max(0, c)
        hi = min(n_samples, c + snippet_after // 2)
        peak = lo + np.argmin(trace[lo:hi])
        if peak - last >= dead_samples:
            refined.append(peak)
            last = peak

    indices = np.array(refined, dtype=int)
    snippet_len = snippet_before + snippet_after
    valid = (indices >= snippet_before) & (indices < n_samples - snippet_after)
    indices = indices[valid]

    waveforms = np.zeros((len(indices), n_channels, snippet_len))
    for i, idx in enumerate(indices):
        waveforms[i] = signal[:, idx - snippet_before : idx + snippet_after]

    return DetectedSpikes(
        indices=indices,
        waveforms=waveforms,
        detection_channel=detection_channel,
        threshold=thresh,
    )
