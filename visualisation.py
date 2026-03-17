"""
visualisation.py — Publication-Quality Matplotlib Figures
==========================================================
Generates separate figures for each analysis stage:
  1. Raw signal + detected spikes overlay
  2. PCA feature space with cluster assignments
  3. Sorted spike waveforms per cluster
  4. Cluster quality metrics dashboard
  5. Raster plots aligned to behavioural events
  6. Peri-stimulus time histograms (PSTHs)
  7. Trial-averaged neural responses synchronised to task structure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ─── Style ───────────────────────────────────────────────────────────────────

PALETTE = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"]
BG_COLOR = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#E2E8F0"
ACCENT_DIM = "#475569"


def _apply_style(fig, axes):
    """Apply dark theme to figure and all axes."""
    fig.patch.set_facecolor(BG_COLOR)
    if not isinstance(axes, np.ndarray):
        axes = [axes] if not isinstance(axes, list) else axes
    else:
        axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, alpha=0.15, color=GRID_COLOR)


# ─── Figure 1: Raw Signal + Spike Detection ─────────────────────────────────

def plot_raw_signal_with_detections(
    signal: np.ndarray,
    fs: int,
    spike_indices: np.ndarray,
    threshold: float,
    detection_channel: int,
    time_window: Tuple[float, float] = (10.0, 12.0),
    save_path: Optional[str] = None,
):
    """Plot raw traces across channels with detected spikes marked."""
    fig, axes = plt.subplots(signal.shape[0], 1, figsize=(16, 8), sharex=True)
    _apply_style(fig, axes)
    fig.suptitle("Raw Signal with Detected Spikes", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.95)

    lo = int(time_window[0] * fs)
    hi = int(time_window[1] * fs)
    t = np.arange(lo, hi) / fs

    for ch, ax in enumerate(axes):
        trace = signal[ch, lo:hi]
        ax.plot(t, trace, color=ACCENT_DIM, linewidth=0.4, alpha=0.8)

        # Mark spikes on this channel
        spk_in = spike_indices[(spike_indices >= lo) & (spike_indices < hi)]
        ax.scatter(spk_in / fs, signal[ch, spk_in], color=PALETTE[0], s=12, zorder=5, marker="v")

        if ch == detection_channel:
            ax.axhline(-threshold, color=PALETTE[1], linestyle="--", alpha=0.6, linewidth=0.8, label=f"Threshold ({threshold:.3f})")
            ax.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

        ax.set_ylabel(f"Ch {ch}", fontsize=9)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 2: PCA Feature Space ────────────────────────────────────────────

def plot_pca_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    explained_variance: np.ndarray,
    save_path: Optional[str] = None,
):
    """Scatter plot of first 2–3 principal components coloured by cluster."""
    n_dims = min(3, features.shape[1])

    if n_dims >= 3:
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax2d = fig.add_subplot(gs[0])
        ax3d = fig.add_subplot(gs[1], projection="3d")
        axes_list = [ax2d, ax3d]
    else:
        fig, ax2d = plt.subplots(1, 1, figsize=(8, 6))
        ax3d = None
        axes_list = [ax2d]

    _apply_style(fig, axes_list)
    fig.suptitle("PCA Feature Space — Cluster Assignments", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.98)

    unique = np.unique(labels)
    for uid in unique:
        mask = labels == uid
        c = PALETTE[uid % len(PALETTE)]
        ax2d.scatter(features[mask, 0], features[mask, 1], c=c, s=4, alpha=0.4, label=f"Unit {uid}")
        ax2d.scatter(centroids[uid, 0], centroids[uid, 1], c=c, s=120, marker="*", edgecolors="white", linewidths=0.5, zorder=10)

    ax2d.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)", fontsize=9)
    ax2d.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)", fontsize=9)
    ax2d.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, markerscale=3)

    if ax3d is not None:
        ax3d.set_facecolor(BG_COLOR)
        ax3d.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax3d.xaxis.label.set_color(TEXT_COLOR)
        ax3d.yaxis.label.set_color(TEXT_COLOR)
        ax3d.zaxis.label.set_color(TEXT_COLOR)
        for uid in unique:
            mask = labels == uid
            c = PALETTE[uid % len(PALETTE)]
            ax3d.scatter(features[mask, 0], features[mask, 1], features[mask, 2], c=c, s=3, alpha=0.3)
        ax3d.set_xlabel(f"PC1", fontsize=8)
        ax3d.set_ylabel(f"PC2", fontsize=8)
        ax3d.set_zlabel(f"PC3", fontsize=8)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 3: Sorted Waveforms ─────────────────────────────────────────────

def plot_sorted_waveforms(
    waveforms: np.ndarray,
    labels: np.ndarray,
    fs: int,
    n_show: int = 100,
    save_path: Optional[str] = None,
):
    """Plot overlaid waveforms per cluster with mean ± std."""
    unique = np.unique(labels)
    n_clusters = len(unique)
    n_channels = waveforms.shape[1]

    fig, axes = plt.subplots(n_clusters, n_channels, figsize=(16, 3 * n_clusters), squeeze=False)
    _apply_style(fig, axes)
    fig.suptitle("Sorted Spike Waveforms by Cluster", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.98)

    snippet_len = waveforms.shape[2]
    t_ms = np.arange(snippet_len) / fs * 1000

    for row, uid in enumerate(unique):
        mask = labels == uid
        cluster_wf = waveforms[mask]
        color = PALETTE[uid % len(PALETTE)]

        for ch in range(n_channels):
            ax = axes[row, ch]
            # Random subset of individual traces
            idx = np.random.choice(len(cluster_wf), min(n_show, len(cluster_wf)), replace=False)
            for i in idx:
                ax.plot(t_ms, cluster_wf[i, ch], color=color, alpha=0.06, linewidth=0.5)

            # Mean ± std
            mean_wf = cluster_wf[:, ch].mean(axis=0)
            std_wf = cluster_wf[:, ch].std(axis=0)
            ax.plot(t_ms, mean_wf, color="white", linewidth=1.5, zorder=5)
            ax.fill_between(t_ms, mean_wf - std_wf, mean_wf + std_wf, color=color, alpha=0.2, zorder=4)

            if ch == 0:
                ax.set_ylabel(f"Unit {uid}", fontsize=9, color=color, fontweight="bold")
            if row == n_clusters - 1:
                ax.set_xlabel("Time (ms)", fontsize=8)
            if row == 0:
                ax.set_title(f"Channel {ch}", fontsize=9, color=TEXT_COLOR)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 4: Quality Metrics ──────────────────────────────────────────────

def plot_quality_metrics(
    quality: Dict[int, Dict],
    save_path: Optional[str] = None,
):
    """Bar charts for ISI violation rate, SNR, firing rate, and spike count."""
    units = sorted(quality.keys())
    metrics = ["n_spikes", "firing_rate_hz", "snr", "isi_violation_rate"]
    titles = ["Spike Count", "Firing Rate (Hz)", "SNR", "ISI Violation Rate"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _apply_style(fig, axes)
    fig.suptitle("Cluster Quality Metrics", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=1.02)

    for ax, metric, title in zip(axes, metrics, titles):
        values = [quality[u][metric] for u in units]
        colors = [PALETTE[u % len(PALETTE)] for u in units]
        bars = ax.bar([f"Unit {u}" for u in units], values, color=colors, edgecolor="none", width=0.6)
        ax.set_title(title, fontsize=10, color=TEXT_COLOR)
        ax.tick_params(axis="x", rotation=0)

        # Highlight bad ISI violations
        if metric == "isi_violation_rate":
            ax.axhline(0.01, color=PALETTE[1], linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(0.98, 0.01, "1% threshold", transform=ax.get_yaxis_transform(),
                    fontsize=7, color=PALETTE[1], alpha=0.7, ha="right", va="bottom")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 5: Raster Plots ─────────────────────────────────────────────────

def plot_raster(
    aligned_data: Dict[int, "AlignedData"],
    event_name: str,
    save_path: Optional[str] = None,
):
    """Raster plot: each row is a trial, each dot is a spike, aligned to event."""
    from behavioural_analysis import AlignedData  # type hint
    units = sorted(aligned_data.keys())
    n_units = len(units)

    fig, axes = plt.subplots(1, n_units, figsize=(4 * n_units, 6), sharey=True, squeeze=False)
    axes = axes[0]
    _apply_style(fig, axes)
    fig.suptitle(f"Raster Plot — Aligned to {event_name.title()}", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.98)

    for ax, uid in zip(axes, units):
        ad = aligned_data[uid]
        color = PALETTE[uid % len(PALETTE)]

        for trial_idx, spikes in enumerate(ad.trial_spike_times):
            ax.scatter(spikes, np.full_like(spikes, trial_idx), color=color, s=1.5, marker="|", linewidths=0.5)

        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Time from event (s)", fontsize=9)
        ax.set_title(f"Unit {uid}", fontsize=10, color=color, fontweight="bold")
        if uid == units[0]:
            ax.set_ylabel("Trial #", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 6: PSTHs ────────────────────────────────────────────────────────

def plot_psth(
    aligned_data: Dict[int, "AlignedData"],
    event_name: str,
    save_path: Optional[str] = None,
):
    """Peri-stimulus time histogram for each unit."""
    units = sorted(aligned_data.keys())
    n_units = len(units)

    fig, axes = plt.subplots(n_units, 1, figsize=(10, 3 * n_units), sharex=True)
    if n_units == 1:
        axes = [axes]
    _apply_style(fig, axes)
    fig.suptitle(f"PSTH — Aligned to {event_name.title()}", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.98)

    for ax, uid in zip(axes, units):
        ad = aligned_data[uid]
        color = PALETTE[uid % len(PALETTE)]

        ax.bar(ad.psth_bins, ad.psth_counts, width=ad.psth_bins[1] - ad.psth_bins[0],
               color=color, alpha=0.7, edgecolor="none")
        ax.fill_between(ad.psth_bins, ad.psth_counts - ad.psth_sem, ad.psth_counts + ad.psth_sem,
                        color=color, alpha=0.2)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(f"Unit {uid}  (n={ad.n_trials} trials)", fontsize=10, color=color, fontweight="bold")

    axes[-1].set_xlabel("Time from event (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig


# ─── Figure 7: Trial-Averaged Responses Across Events ───────────────────────

def plot_trial_averaged_across_events(
    all_aligned: Dict[str, Dict[int, "AlignedData"]],
    save_path: Optional[str] = None,
):
    """
    Overlay trial-averaged firing rates across stimulus, response, reward events.
    One subplot per unit, overlaid lines per event type.
    """
    event_names = list(all_aligned.keys())
    # Get units from first event
    units = sorted(all_aligned[event_names[0]].keys())
    n_units = len(units)
    event_colors = {"stimulus": PALETTE[0], "response": PALETTE[2], "reward": PALETTE[3]}

    fig, axes = plt.subplots(n_units, 1, figsize=(10, 3 * n_units), sharex=True)
    if n_units == 1:
        axes = [axes]
    _apply_style(fig, axes)
    fig.suptitle("Trial-Averaged Firing Rates — All Events", fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.98)

    for ax, uid in zip(axes, units):
        for ev_name in event_names:
            ad = all_aligned[ev_name][uid]
            c = event_colors.get(ev_name, PALETTE[5])
            ax.plot(ad.psth_bins, ad.psth_counts, color=c, linewidth=1.5, label=ev_name.title(), alpha=0.9)
            ax.fill_between(ad.psth_bins, ad.psth_counts - ad.psth_sem, ad.psth_counts + ad.psth_sem,
                            color=c, alpha=0.12)

        ax.axvline(0, color="white", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(f"Unit {uid}", fontsize=10, color=PALETTE[uid % len(PALETTE)], fontweight="bold")
        if uid == units[0]:
            ax.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper right")

    axes[-1].set_xlabel("Time from event (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    return fig
