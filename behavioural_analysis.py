"""
behavioural_analysis.py — Event-Aligned Neural Analysis
========================================================
Maps sorted neural firing to behavioural task events:
  • Peri-stimulus time histograms (PSTHs)
  • Raster plots (per-trial spike times relative to events)
  • Trial-averaged neural responses
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AlignedData:
    """Spike data aligned to a behavioural event."""
    event_name: str
    unit_id: int
    trial_spike_times: List[np.ndarray]  # list of arrays, one per trial (in seconds relative to event)
    psth_bins: np.ndarray                # bin centres
    psth_counts: np.ndarray              # firing rate per bin (Hz)
    psth_sem: np.ndarray                 # standard error of the mean
    n_trials: int


def align_spikes_to_events(
    spike_indices: np.ndarray,
    labels: np.ndarray,
    event_times: np.ndarray,
    fs: int,
    window: Tuple[float, float] = (-0.5, 1.0),
    bin_width: float = 0.025,
) -> Dict[int, AlignedData]:
    """
    Align sorted spikes to event times and compute PSTHs.

    Parameters
    ----------
    spike_indices : sample indices of all detected spikes
    labels : cluster label per spike
    event_times : event timestamps in seconds
    fs : sampling rate
    window : (pre, post) in seconds around event
    bin_width : PSTH bin width in seconds

    Returns
    -------
    Dictionary mapping unit_id → AlignedData
    """
    spike_times = spike_indices / fs  # convert to seconds
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    unique_units = np.unique(labels)
    results = {}

    for unit_id in unique_units:
        unit_mask = labels == unit_id
        unit_times = spike_times[unit_mask]

        trial_spikes = []
        trial_counts = []

        for ev_t in event_times:
            # Find spikes within window
            rel = unit_times - ev_t
            in_window = rel[(rel >= window[0]) & (rel <= window[1])]
            trial_spikes.append(in_window)

            # Histogram for this trial
            counts, _ = np.histogram(in_window, bins=bins)
            trial_counts.append(counts / bin_width)  # convert to Hz

        trial_counts = np.array(trial_counts)  # (n_trials, n_bins)
        mean_rate = trial_counts.mean(axis=0)
        sem_rate = trial_counts.std(axis=0) / np.sqrt(len(event_times))

        results[int(unit_id)] = AlignedData(
            event_name="",  # filled in by caller
            unit_id=int(unit_id),
            trial_spike_times=trial_spikes,
            psth_bins=bin_centres,
            psth_counts=mean_rate,
            psth_sem=sem_rate,
            n_trials=len(event_times),
        )

    return results


def compute_trial_averaged_waveforms(
    waveforms: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute mean ± std waveform per cluster.

    Returns
    -------
    unit_id → (mean_waveform, std_waveform)
    Both shapes: (n_channels, snippet_len)
    """
    results = {}
    for uid in np.unique(labels):
        mask = labels == uid
        cluster_wfs = waveforms[mask]
        results[int(uid)] = (cluster_wfs.mean(axis=0), cluster_wfs.std(axis=0))
    return results
