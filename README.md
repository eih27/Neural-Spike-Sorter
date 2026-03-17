# Neural Spike Sorting & Behavioural Analysis Tool

A complete Python pipeline for extracellular electrophysiology analysis: from simulated recordings through spike detection, sorting, and behavioural event alignment.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)

---

## Pipeline Overview

```
Simulated Recording → Bandpass Filter → Spike Detection → PCA + K-Means → Behavioural Alignment → Figures
```

### 1. Data Simulation (`simulate_data.py`)
- 4-channel extracellular recordings (30 kHz, 120s)
- 4 distinct neuron waveform templates with spatial decay across channels
- Poisson spike trains with refractory period enforcement
- Pink (1/f) noise + motion artifacts
- Behavioural event timestamps: stimulus → response → reward

### 2. Spike Detection (`spike_detection.py`)
- **Threshold-based**: MAD (median absolute deviation) with dead-time enforcement
- **Wavelet-based**: Stationary Wavelet Transform (SWT) using detail coefficients
- Automatic detection channel selection (highest variance)
- Multi-channel waveform snippet extraction

### 3. Spike Sorting (`spike_sorting.py`)
- PCA dimensionality reduction (95% variance threshold)
- K-means clustering with automatic K selection (silhouette score)
- Per-cluster quality metrics: ISI violation rate, SNR, firing rate

### 4. Behavioural Analysis (`behavioural_analysis.py`)
- Event-aligned spike rasters
- Peri-stimulus time histograms (PSTHs) with SEM
- Trial-averaged firing rates across stimulus, response, and reward events

### 5. Visualisation (`visualisation.py`)
Seven publication-quality figures with a dark theme:

| # | Figure | Description |
|---|--------|-------------|
| 1 | Raw Signal + Detections | Filtered traces with spike markers and threshold |
| 2 | PCA Feature Space | 2D/3D scatter of principal components by cluster |
| 3 | Sorted Waveforms | Overlaid waveforms per unit per channel with mean±std |
| 4 | Quality Metrics | Bar charts: spike count, firing rate, SNR, ISI violations |
| 5 | Raster Plot | Trial-by-trial spike times aligned to stimulus onset |
| 6 | PSTH | Peri-stimulus time histograms per unit |
| 7 | Trial-Averaged Responses | Firing rates aligned to all three event types |

---

## Quick Start

```bash
# Clone and install
git clone <your-repo-url>
cd neural-spike-sorter
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Custom output directory and seed
python main.py --output-dir ./results --seed 123
```

All figures are saved as high-resolution PNGs in the `output/` directory.

---

## Hosting

### GitHub Pages (simplest)
1. Push to GitHub
2. Add output PNGs to a `docs/` folder or use a simple `index.html` gallery
3. Enable GitHub Pages in repo settings

### Streamlit (interactive)
```bash
pip install streamlit
# Wrap main.py in a Streamlit app for parameter tuning
streamlit run app.py
```

### Static Site
The output PNGs can be embedded in any static site (Hugo, Jekyll, plain HTML).

---

## Project Structure

```
neural-spike-sorter/
├── main.py                  # Entry point — runs full pipeline
├── simulate_data.py         # Electrophysiology data generator
├── spike_detection.py       # Threshold + wavelet spike detection
├── spike_sorting.py         # PCA + K-means clustering
├── behavioural_analysis.py  # Event alignment, PSTHs, rasters
├── visualisation.py         # All Matplotlib figures
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── output/                  # Generated figures (after running)
```

---

## Configuration

All simulation parameters are in `SimConfig` (in `simulate_data.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_s` | 120.0 | Recording length (seconds) |
| `fs` | 30,000 | Sampling rate (Hz) |
| `n_channels` | 4 | Number of extracellular channels |
| `n_units` | 4 | Number of distinct neurons |
| `noise_level` | 0.15 | Background noise RMS |
| `n_trials` | 60 | Number of behavioural trials |
| `baseline_rates` | (5, 12, 8, 3) | Baseline firing rates (Hz) |
| `stim_rate_gain` | (3, 1.5, 2.5, 4) | Rate multiplier during stimulation |

---

## License

MIT
