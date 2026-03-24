#!/usr/bin/env python3
"""
app.py — Streamlit Interactive Neural Spike Sorting Dashboard
==============================================================
Interactive web app for the full spike sorting pipeline.
Visitors can tweak simulation parameters and watch the pipeline run.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO
import time

from simulate_data import generate, SimConfig
from spike_detection import bandpass_filter, detect_threshold, detect_wavelet
from spike_sorting import sort_spikes
from behavioural_analysis import align_spikes_to_events, compute_trial_averaged_waveforms

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Spike Sorter",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Style Constants ─────────────────────────────────────────────────────────

PALETTE = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"]
BG_COLOR = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#E2E8F0"
ACCENT_DIM = "#475569"


def apply_style(fig, axes):
    """Apply dark theme."""
    fig.patch.set_facecolor(BG_COLOR)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(BG_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
            ax.spines['bottom'].set_color(GRID_COLOR)
            ax.spines['left'].set_color(GRID_COLOR)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)
            ax.title.set_color(TEXT_COLOR)


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 1rem;
        border-left: 3px solid #3B82F6;
        padding-left: 0.8rem;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3B82F6;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    .tech-badge {
        display: inline-block;
        background: #1E293B;
        color: #94A3B8;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin: 0.1rem;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Controls ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    st.markdown("**Simulation**")
    duration = st.slider("Recording duration (s)", 30, 300, 120, step=30)
    n_units = st.slider("Number of neurons", 2, 6, 4)
    noise_level = st.slider("Noise level", 0.05, 0.40, 0.15, step=0.05)
    n_trials = st.slider("Behavioural trials", 20, 120, 60, step=10)

    st.markdown("---")
    st.markdown("**Detection**")
    detection_method = st.selectbox("Method", ["Threshold (MAD)", "Wavelet (SWT)"])
    n_sigma = st.slider("Detection threshold (σ)", 2.5, 6.0, 4.0, step=0.5)

    st.markdown("---")
    st.markdown("**Sorting**")
    max_clusters = st.slider("Max clusters (K)", 3, 12, 8)

    st.markdown("---")
    seed = st.number_input("Random seed", value=42, step=1)

    run_button = st.button("🚀 Run Pipeline", width="stretch")

    st.markdown("---")
    st.markdown(
        '<span class="tech-badge">Python</span>'
        '<span class="tech-badge">NumPy</span>'
        '<span class="tech-badge">SciPy</span>'
        '<span class="tech-badge">scikit-learn</span>'
        '<span class="tech-badge">Matplotlib</span>',
        unsafe_allow_html=True,
    )


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Neural Spike Sorting & Behavioural Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'End-to-end pipeline: simulated electrophysiology → spike detection → PCA + K-means sorting → behavioural event alignment'
    '</div>',
    unsafe_allow_html=True,
)


# ─── Pipeline Execution ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(duration, n_units, noise_level, n_trials, detection_method, n_sigma, max_clusters, seed):
    """Run the full pipeline and cache results."""
    base_rates = (5.0, 12.0, 8.0, 3.0, 7.0, 15.0)[:n_units]
    stim_gains = (3.0, 1.5, 2.5, 4.0, 2.0, 1.8)[:n_units]

    cfg = SimConfig(
        duration_s=float(duration),
        n_units=n_units,
        noise_level=noise_level,
        n_trials=n_trials,
        baseline_rates=base_rates,
        stim_rate_gain=stim_gains,
    )

    # 1. Simulate
    data = generate(cfg, seed=int(seed))

    # 2. Filter
    filtered = bandpass_filter(data.raw_signal, cfg.fs, low=300, high=6000)

    # 3. Detect
    if detection_method == "Threshold (MAD)":
        detected = detect_threshold(filtered, cfg.fs, n_sigma=n_sigma)
    else:
        detected = detect_wavelet(filtered, cfg.fs, n_sigma=n_sigma)

    # 4. Sort
    sorting = sort_spikes(
        detected.waveforms, detected.indices, cfg.fs,
        k_range=(2, max_clusters), seed=int(seed),
    )

    # 5. Align to events
    all_aligned = {}
    for event_name in ["stimulus", "response", "reward"]:
        aligned = align_spikes_to_events(
            detected.indices, sorting.labels,
            data.events[event_name], cfg.fs,
            window=(-0.5, 1.0), bin_width=0.025,
        )
        for uid in aligned:
            aligned[uid].event_name = event_name
        all_aligned[event_name] = aligned

    # 6. Trial-averaged waveforms
    avg_waveforms = compute_trial_averaged_waveforms(
        detected.waveforms, detected.indices, sorting.labels,
        data.events["stimulus"], cfg.fs, window=(-0.5, 1.0),
    )

    return {
        "cfg": cfg, "data": data, "filtered": filtered,
        "detected": detected, "sorting": sorting,
        "all_aligned": all_aligned, "avg_waveforms": avg_waveforms,
    }


# ─── Run on Button Press ────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_button or st.session_state["results"] is None:
    with st.spinner("Running pipeline..."):
        progress = st.progress(0)
        for i in range(5):
            progress.progress((i + 1) * 20)
            time.sleep(0.2)
        results = run_pipeline(duration, n_units, noise_level, n_trials,
                               detection_method, n_sigma, max_clusters, seed)
        progress.progress(100)
        time.sleep(0.3)
        progress.empty()
        st.session_state["results"] = results

results = st.session_state["results"]
cfg = results["cfg"]
data = results["data"]
filtered = results["filtered"]
detected = results["detected"]
sorting = results["sorting"]
all_aligned = results["all_aligned"]
avg_waveforms = results["avg_waveforms"]


# ─── Summary Metrics ─────────────────────────────────────────────────────────

total_gt_spikes = sum(len(v) for v in data.spike_trains.values())

cols = st.columns(6)
metric_data = [
    (f"{cfg.duration_s:.0f}s", "Recording"),
    (f"{cfg.n_channels}", "Channels"),
    (f"{len(detected.indices):,}", "Detected Spikes"),
    (f"{sorting.n_clusters}", "Clusters Found"),
    (f"{len(data.events['stimulus'])}", "Trials"),
    (f"{sorting.explained_variance.sum()*100:.0f}%", "PCA Variance"),
]
for col, (val, label) in zip(cols, metric_data):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")


# ─── Tabbed Figures ──────────────────────────────────────────────────────────

tab_raw, tab_pca, tab_wf, tab_quality, tab_raster, tab_psth, tab_avg = st.tabs([
    "📊 Raw Signal",
    "🔬 PCA Clusters",
    "〰️ Waveforms",
    "✅ Quality",
    "📍 Rasters",
    "📈 PSTHs",
    "🧪 Trial-Averaged",
])


# ── Tab 1: Raw Signal ───────────────────────────────────────────────────────

with tab_raw:
    st.markdown('<div class="section-header">Filtered Signal with Detected Spikes</div>', unsafe_allow_html=True)

    t_start = st.slider("Start time (s)", 0.0, float(cfg.duration_s - 2), 10.0, step=0.5, key="raw_t")
    t_window = 2.0
    lo = int(t_start * cfg.fs)
    hi = int((t_start + t_window) * cfg.fs)
    t = np.arange(lo, hi) / cfg.fs

    fig, axes = plt.subplots(cfg.n_channels, 1, figsize=(14, 7), sharex=True)
    apply_style(fig, axes)
    fig.suptitle("Bandpass-Filtered Signal + Spike Detections", fontsize=13,
                 color=TEXT_COLOR, fontweight="bold", y=0.95)

    for ch, ax in enumerate(axes):
        trace = filtered[ch, lo:hi]
        ax.plot(t, trace, color=ACCENT_DIM, linewidth=0.4, alpha=0.8)

        spk_in = detected.indices[(detected.indices >= lo) & (detected.indices < hi)]
        ax.scatter(spk_in / cfg.fs, filtered[ch, spk_in],
                   color=PALETTE[0], s=10, zorder=5, marker="v")

        if ch == detected.detection_channel:
            ax.axhline(-detected.threshold, color=PALETTE[1], linestyle="--",
                       alpha=0.6, linewidth=0.8)

        ax.set_ylabel(f"Ch {ch}", fontsize=9)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 2: PCA Clusters ─────────────────────────────────────────────────────

with tab_pca:
    st.markdown('<div class="section-header">PCA Feature Space — Cluster Assignments</div>', unsafe_allow_html=True)

    features = sorting.pca_features
    labels = sorting.labels
    n_dims = min(3, features.shape[1])

    fig = plt.figure(figsize=(14, 6))
    if n_dims >= 3:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax2d = fig.add_subplot(gs[0])
        ax3d = fig.add_subplot(gs[1], projection="3d")
        all_axes = [ax2d, ax3d]
    else:
        ax2d = fig.add_subplot(111)
        ax3d = None
        all_axes = [ax2d]

    apply_style(fig, all_axes)
    fig.suptitle("PCA Feature Space", fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.98)

    for uid in np.unique(labels):
        mask = labels == uid
        c = PALETTE[uid % len(PALETTE)]
        ax2d.scatter(features[mask, 0], features[mask, 1], c=c, s=4, alpha=0.4, label=f"Unit {uid}")
        ax2d.scatter(sorting.centroids[uid, 0], sorting.centroids[uid, 1],
                     c=c, s=120, marker="*", edgecolors="white", linewidths=0.5, zorder=10)

    ax2d.set_xlabel(f"PC1 ({sorting.explained_variance[0]*100:.1f}%)", fontsize=9)
    ax2d.set_ylabel(f"PC2 ({sorting.explained_variance[1]*100:.1f}%)", fontsize=9)
    ax2d.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR, markerscale=3)

    if ax3d is not None:
        ax3d.set_facecolor(BG_COLOR)
        ax3d.tick_params(colors=TEXT_COLOR, labelsize=7)
        for uid in np.unique(labels):
            mask = labels == uid
            c = PALETTE[uid % len(PALETTE)]
            ax3d.scatter(features[mask, 0], features[mask, 1], features[mask, 2], c=c, s=3, alpha=0.3)
        ax3d.set_xlabel("PC1", fontsize=8, color=TEXT_COLOR)
        ax3d.set_ylabel("PC2", fontsize=8, color=TEXT_COLOR)
        ax3d.set_zlabel("PC3", fontsize=8, color=TEXT_COLOR)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 3: Sorted Waveforms ─────────────────────────────────────────────────

with tab_wf:
    st.markdown('<div class="section-header">Sorted Spike Waveforms by Cluster</div>', unsafe_allow_html=True)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_channels = detected.waveforms.shape[1]
    snippet_len = detected.waveforms.shape[2]
    t_ms = np.arange(snippet_len) / cfg.fs * 1000

    fig, axes = plt.subplots(n_clusters, n_channels, figsize=(16, 2.5 * n_clusters), squeeze=False)
    apply_style(fig, axes)
    fig.suptitle("Sorted Waveforms", fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.99)

    for row, uid in enumerate(unique_labels):
        mask = labels == uid
        cluster_wf = detected.waveforms[mask]
        color = PALETTE[uid % len(PALETTE)]

        for ch in range(n_channels):
            ax = axes[row, ch]
            n_show = min(80, len(cluster_wf))
            idx = np.random.default_rng(42).choice(len(cluster_wf), n_show, replace=False)
            for i in idx:
                ax.plot(t_ms, cluster_wf[i, ch], color=color, alpha=0.06, linewidth=0.5)

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

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 4: Quality Metrics ──────────────────────────────────────────────────

with tab_quality:
    st.markdown('<div class="section-header">Cluster Quality Metrics</div>', unsafe_allow_html=True)

    quality = sorting.quality
    units = sorted(quality.keys())

    # Summary table
    quality_cols = st.columns(len(units))
    for col, uid in zip(quality_cols, units):
        q = quality[uid]
        color = PALETTE[uid % len(PALETTE)]
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">Unit {uid}</div>'
            f'<div class="metric-label">Spikes: {q["n_spikes"]:,}<br>'
            f'Rate: {q["firing_rate"]:.1f} Hz<br>'
            f'SNR: {q["snr"]:.1f}<br>'
            f'ISI viol: {q["isi_violation_rate"]*100:.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Bar charts
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    apply_style(fig, axes)
    fig.suptitle("Quality Metrics", fontsize=13, color=TEXT_COLOR, fontweight="bold", y=1.02)

    metrics = ["n_spikes", "firing_rate", "snr", "isi_violation_rate"]
    titles = ["Spike Count", "Firing Rate (Hz)", "SNR", "ISI Violation Rate"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [quality[u][metric] for u in units]
        colors = [PALETTE[u % len(PALETTE)] for u in units]
        bars = ax.bar([f"U{u}" for u in units], vals, color=colors, alpha=0.8, edgecolor="none")
        ax.set_title(title, fontsize=10, color=TEXT_COLOR, fontweight="bold")
        for bar, v in zip(bars, vals):
            fmt = f"{v:,.0f}" if metric == "n_spikes" else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7, color=TEXT_COLOR)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 5: Rasters ──────────────────────────────────────────────────────────

with tab_raster:
    st.markdown('<div class="section-header">Spike Rasters Aligned to Events</div>', unsafe_allow_html=True)

    event_choice_r = st.selectbox("Align to event:", ["stimulus", "response", "reward"], key="raster_ev")
    aligned_r = all_aligned[event_choice_r]
    units_r = sorted(aligned_r.keys())

    fig, axes = plt.subplots(1, len(units_r), figsize=(4 * len(units_r), 6), sharey=True)
    if len(units_r) == 1:
        axes = [axes]
    apply_style(fig, axes)
    fig.suptitle(f"Raster — {event_choice_r.title()}", fontsize=13,
                 color=TEXT_COLOR, fontweight="bold", y=0.98)

    for ax, uid in zip(axes, units_r):
        ad = aligned_r[uid]
        color = PALETTE[uid % len(PALETTE)]
        for trial_idx, spikes in enumerate(ad.raster):
            ax.scatter(spikes, np.full_like(spikes, trial_idx),
                       color=color, s=2, marker="|", linewidths=0.6)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_title(f"Unit {uid}", fontsize=10, color=color, fontweight="bold")
        if uid == units_r[0]:
            ax.set_ylabel("Trial #", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 6: PSTHs ────────────────────────────────────────────────────────────

with tab_psth:
    st.markdown('<div class="section-header">Peri-Stimulus Time Histograms</div>', unsafe_allow_html=True)

    event_choice_p = st.selectbox("Align to event:", ["stimulus", "response", "reward"], key="psth_ev")
    aligned_p = all_aligned[event_choice_p]
    units_p = sorted(aligned_p.keys())

    fig, axes = plt.subplots(len(units_p), 1, figsize=(10, 2.5 * len(units_p)), sharex=True)
    if len(units_p) == 1:
        axes = [axes]
    apply_style(fig, axes)
    fig.suptitle(f"PSTH — {event_choice_p.title()}", fontsize=13,
                 color=TEXT_COLOR, fontweight="bold", y=0.99)

    for ax, uid in zip(axes, units_p):
        ad = aligned_p[uid]
        color = PALETTE[uid % len(PALETTE)]
        bw = ad.psth_bins[1] - ad.psth_bins[0]
        ax.bar(ad.psth_bins, ad.psth_counts, width=bw, color=color, alpha=0.7, edgecolor="none")
        ax.fill_between(ad.psth_bins, ad.psth_counts - ad.psth_sem,
                        ad.psth_counts + ad.psth_sem, color=color, alpha=0.2)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(f"Unit {uid} (n={ad.n_trials})", fontsize=10, color=color, fontweight="bold")

    axes[-1].set_xlabel("Time from event (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ── Tab 7: Trial-Averaged ───────────────────────────────────────────────────

with tab_avg:
    st.markdown('<div class="section-header">Trial-Averaged Firing Rates — All Events</div>', unsafe_allow_html=True)

    units_a = sorted(all_aligned["stimulus"].keys())
    event_colors = {"stimulus": PALETTE[0], "response": PALETTE[2], "reward": PALETTE[3]}

    fig, axes = plt.subplots(len(units_a), 1, figsize=(10, 2.5 * len(units_a)), sharex=True)
    if len(units_a) == 1:
        axes = [axes]
    apply_style(fig, axes)
    fig.suptitle("Trial-Averaged Firing Rates", fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.99)

    for ax, uid in zip(axes, units_a):
        for ev_name in ["stimulus", "response", "reward"]:
            ad = all_aligned[ev_name][uid]
            c = event_colors[ev_name]
            ax.plot(ad.psth_bins, ad.psth_counts, color=c, linewidth=1.5,
                    label=ev_name.title(), alpha=0.9)
            ax.fill_between(ad.psth_bins, ad.psth_counts - ad.psth_sem,
                            ad.psth_counts + ad.psth_sem, color=c, alpha=0.12)
        ax.axvline(0, color="white", linewidth=0.6, linestyle=":", alpha=0.5)
        ax.set_ylabel("Rate (Hz)", fontsize=9)
        ax.set_title(f"Unit {uid}", fontsize=10, color=PALETTE[uid % len(PALETTE)], fontweight="bold")
        if uid == units_a[0]:
            ax.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                      labelcolor=TEXT_COLOR, loc="upper right")

    axes[-1].set_xlabel("Time from event (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:0.8rem;padding:1rem 0;">'
    'Neural Spike Sorting & Behavioural Analysis Tool · '
    'Built with Python, NumPy, SciPy, scikit-learn, Matplotlib & Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
