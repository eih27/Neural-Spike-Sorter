#!/usr/bin/env python3
"""
main.py — Neural Spike Sorting & Behavioural Analysis Pipeline
================================================================
End-to-end pipeline:
  1. Simulate extracellular recordings with behavioural events
  2. Bandpass filter + spike detection (threshold & wavelet)
  3. PCA feature extraction + K-means spike sorting
  4. Cluster quality metrics (ISI violations, SNR)
  5. Behavioural event alignment: rasters, PSTHs, trial-averaged responses
  6. Generate all publication-quality figures

Usage:
    python main.py
    python main.py --output-dir ./my_figures --seed 123
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# ─── Pipeline Modules ────────────────────────────────────────────────────────
from simulate_data import generate, SimConfig
from spike_detection import bandpass_filter, detect_threshold, detect_wavelet
from spike_sorting import sort_spikes
from behavioural_analysis import align_spikes_to_events, compute_trial_averaged_waveforms
from visualisation import (
    plot_raw_signal_with_detections,
    plot_pca_clusters,
    plot_sorted_waveforms,
    plot_quality_metrics,
    plot_raster,
    plot_psth,
    plot_trial_averaged_across_events,
)


def run_pipeline(output_dir: str = "output", seed: int = 42):
    """Execute the full analysis pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Neural Spike Sorting & Behavioural Analysis Tool")
    print("=" * 70)

    # ── Step 1: Simulate Data ────────────────────────────────────────────
    print("\n[1/6] Generating simulated electrophysiology data...")
    cfg = SimConfig()
    data = generate(cfg, seed=seed)

    total_spikes = sum(len(v) for v in data.spike_trains.values())
    print(f"       → {cfg.duration_s}s recording, {cfg.n_channels} channels, "
          f"{cfg.n_units} ground-truth units")
    print(f"       → {total_spikes} total spikes, {len(data.events['stimulus'])} behavioural trials")

    # ── Step 2: Bandpass Filter + Spike Detection ────────────────────────
    print("\n[2/6] Filtering & detecting spikes...")
    filtered = bandpass_filter(data.raw_signal, cfg.fs, low=300, high=6000)

    det_thresh = detect_threshold(filtered, cfg.fs, n_sigma=4.0)
    det_wavelet = detect_wavelet(filtered, cfg.fs, n_sigma=3.5)

    print(f"       → Threshold detection: {len(det_thresh.indices)} spikes "
          f"(channel {det_thresh.detection_channel}, thresh={det_thresh.threshold:.4f})")
    print(f"       → Wavelet detection:   {len(det_wavelet.indices)} spikes")

    # Use threshold detection as primary (more predictable)
    detected = det_thresh

    # ── Step 3: Spike Sorting ────────────────────────────────────────────
    print("\n[3/6] Sorting spikes (PCA + K-means)...")
    sorting = sort_spikes(
        detected.waveforms,
        detected.spike_indices if hasattr(detected, "spike_indices") else detected.indices,
        cfg.fs,
    )
    print(f"       → Found {sorting.n_clusters} clusters")
    print(f"       → PCA: {len(sorting.explained_variance)} components, "
          f"{sorting.explained_variance.sum()*100:.1f}% variance explained")

    for uid, q in sorting.quality.items():
        print(f"       → Unit {uid}: {q['n_spikes']} spikes, "
              f"FR={q['firing_rate_hz']}Hz, SNR={q['snr']}, "
              f"ISI viol={q['isi_violation_rate']*100:.2f}%")

    # ── Step 4: Behavioural Analysis ─────────────────────────────────────
    print("\n[4/6] Aligning spikes to behavioural events...")
    all_aligned = {}
    for event_name in ["stimulus", "response", "reward"]:
        aligned = align_spikes_to_events(
            detected.indices,
            sorting.labels,
            data.events[event_name],
            cfg.fs,
            window=(-0.5, 1.0),
            bin_width=0.025,
        )
        for uid in aligned:
            aligned[uid].event_name = event_name
        all_aligned[event_name] = aligned
        print(f"       → Aligned to {event_name}: {len(data.events[event_name])} events")

    # ── Step 5: Trial-Averaged Waveforms ─────────────────────────────────
    print("\n[5/6] Computing trial-averaged waveforms...")
    avg_waveforms = compute_trial_averaged_waveforms(detected.waveforms, sorting.labels)

    # ── Step 6: Generate All Figures ─────────────────────────────────────
    print("\n[6/6] Generating figures...")
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving

    fig1 = plot_raw_signal_with_detections(
        filtered, cfg.fs, detected.indices, detected.threshold,
        detected.detection_channel, time_window=(10.0, 12.0),
        save_path=str(out / "01_raw_signal_detections.png"),
    )
    print("       → 01_raw_signal_detections.png")

    fig2 = plot_pca_clusters(
        sorting.pca_features, sorting.labels, sorting.centroids,
        sorting.explained_variance,
        save_path=str(out / "02_pca_clusters.png"),
    )
    print("       → 02_pca_clusters.png")

    fig3 = plot_sorted_waveforms(
        detected.waveforms, sorting.labels, cfg.fs,
        save_path=str(out / "03_sorted_waveforms.png"),
    )
    print("       → 03_sorted_waveforms.png")

    fig4 = plot_quality_metrics(
        sorting.quality,
        save_path=str(out / "04_quality_metrics.png"),
    )
    print("       → 04_quality_metrics.png")

    fig5 = plot_raster(
        all_aligned["stimulus"], "stimulus",
        save_path=str(out / "05_raster_stimulus.png"),
    )
    print("       → 05_raster_stimulus.png")

    fig6 = plot_psth(
        all_aligned["stimulus"], "stimulus",
        save_path=str(out / "06_psth_stimulus.png"),
    )
    print("       → 06_psth_stimulus.png")

    fig7 = plot_trial_averaged_across_events(
        all_aligned,
        save_path=str(out / "07_trial_averaged_all_events.png"),
    )
    print("       → 07_trial_averaged_all_events.png")

    # Close all figures
    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\n{'=' * 70}")
    print(f"  ✓ Pipeline complete! {7} figures saved to: {out.resolve()}")
    print(f"{'=' * 70}")

    return {
        "data": data,
        "filtered": filtered,
        "detected": detected,
        "sorting": sorting,
        "aligned": all_aligned,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Spike Sorting Pipeline")
    parser.add_argument("--output-dir", default="output", help="Directory for output figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(output_dir=args.output_dir, seed=args.seed)
