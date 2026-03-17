"""
simulate_data.py — Simulated Extracellular Electrophysiology Data Generator
=============================================================================
Generates realistic multi-channel recordings with:
  • 4 extracellular channels
  • 4 distinct neuron waveform templates (different shapes & amplitudes)
  • Poisson-distributed spike trains with refractory periods
  • Pink (1/f) background noise + occasional motion artifacts
  • Behavioural event timestamps: stimulus onset, response, reward
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All tuneable simulation parameters in one place."""
    duration_s: float = 120.0          # total recording length (seconds)
    fs: int = 30_000                   # sampling rate (Hz)
    n_channels: int = 4                # number of extracellular channels
    n_units: int = 4                   # number of distinct neurons
    noise_level: float = 0.15          # RMS of background noise (a.u.)
    artifact_prob: float = 0.002       # probability of motion artifact per sample
    refractory_ms: float = 1.5         # absolute refractory period (ms)

    # Behavioural task parameters
    n_trials: int = 60                 # number of behavioural trials
    iti_range: Tuple[float, float] = (1.0, 2.5)   # inter-trial interval (s)
    response_delay: Tuple[float, float] = (0.2, 0.6)
    reward_delay: Tuple[float, float] = (0.1, 0.3)

    # Firing-rate modulation around events
    baseline_rates: Tuple[float, ...] = (5.0, 12.0, 8.0, 3.0)  # Hz per unit
    stim_rate_gain: Tuple[float, ...] = (3.0, 1.5, 2.5, 4.0)   # multiplier during stim


@dataclass
class SimulatedData:
    """Container for all generated data."""
    raw_signal: np.ndarray              # (n_channels, n_samples)
    spike_trains: Dict[int, np.ndarray] # unit_id → sample indices
    waveform_templates: Dict[int, np.ndarray]  # unit_id → (n_channels, n_waveform_samples)
    events: Dict[str, np.ndarray]       # event_name → timestamps in seconds
    config: SimConfig = field(default_factory=SimConfig)


# ─── Waveform Templates ─────────────────────────────────────────────────────

def _make_waveform_templates(cfg: SimConfig, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """
    Create distinct spike waveform templates for each unit across channels.
    Each unit has a characteristic shape (varied peak widths, asymmetry, amplitude).
    Spatial footprint decays across channels from a 'best' channel.
    """
    n_pts = int(0.002 * cfg.fs)  # 2 ms waveform window
    t = np.linspace(0, 1, n_pts)
    templates = {}

    # Base waveform shapes (parametric)
    shapes = [
        lambda t: -2.5 * np.sin(np.pi * t) ** 2 + 0.6 * np.sin(2 * np.pi * t),           # sharp negative
        lambda t: -1.8 * np.sin(np.pi * t) ** 1.5 + 1.0 * np.sin(1.5 * np.pi * t) ** 2,  # broad biphasic
        lambda t: -3.0 * np.exp(-((t - 0.35) ** 2) / 0.01) + 0.8 * np.exp(-((t - 0.7) ** 2) / 0.02),  # narrow Gaussian
        lambda t: -2.0 * np.sin(np.pi * t) ** 3 + 0.4 * np.sin(3 * np.pi * t),            # triphasic
    ]

    for unit_id in range(cfg.n_units):
        base = shapes[unit_id % len(shapes)](t)
        # Amplitude jitter
        base *= (0.8 + 0.4 * rng.random())
        # Spatial decay: pick a 'best' channel, decay on others
        best_ch = unit_id % cfg.n_channels
        spatial = np.zeros((cfg.n_channels, n_pts))
        for ch in range(cfg.n_channels):
            decay = np.exp(-0.8 * abs(ch - best_ch))
            phase_shift = int(rng.integers(0, max(1, n_pts // 10)))
            shifted = np.roll(base, phase_shift)
            spatial[ch] = shifted * decay * (0.9 + 0.2 * rng.random())
        templates[unit_id] = spatial

    return templates


# ─── Spike Train Generation ─────────────────────────────────────────────────

def _generate_spike_trains(
    cfg: SimConfig,
    events: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """
    Generate Poisson spike trains for each unit with:
      - refractory period enforcement
      - firing-rate modulation around stimulus events
    """
    n_samples = int(cfg.duration_s * cfg.fs)
    refractory_samples = int(cfg.refractory_ms / 1000.0 * cfg.fs)
    stim_times = events["stimulus"]

    spike_trains = {}
    for unit_id in range(cfg.n_units):
        # Build time-varying rate
        rate = np.full(n_samples, cfg.baseline_rates[unit_id])

        # Modulate rate around stimuli (±200 ms window)
        window_samples = int(0.2 * cfg.fs)
        for st in stim_times:
            centre = int(st * cfg.fs)
            lo = max(0, centre - window_samples // 4)
            hi = min(n_samples, centre + window_samples)
            rate[lo:hi] *= cfg.stim_rate_gain[unit_id]

        # Poisson sampling
        prob = rate / cfg.fs
        spikes_bool = rng.random(n_samples) < prob

        # Enforce refractory period
        spike_indices = []
        last_spike = -refractory_samples - 1
        for i in np.where(spikes_bool)[0]:
            if i - last_spike > refractory_samples:
                spike_indices.append(i)
                last_spike = i

        spike_trains[unit_id] = np.array(spike_indices, dtype=int)

    return spike_trains


# ─── Behavioural Events ─────────────────────────────────────────────────────

def _generate_events(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate stimulus → response → reward event sequences."""
    stim_times = []
    t = rng.uniform(*cfg.iti_range)
    for _ in range(cfg.n_trials):
        stim_times.append(t)
        t += rng.uniform(*cfg.iti_range) + rng.uniform(*cfg.response_delay) + rng.uniform(*cfg.reward_delay)
        if t >= cfg.duration_s - 1.0:
            break

    stim_times = np.array(stim_times)
    response_times = stim_times + rng.uniform(*cfg.response_delay, size=len(stim_times))
    reward_times = response_times + rng.uniform(*cfg.reward_delay, size=len(stim_times))

    return {
        "stimulus": stim_times,
        "response": response_times,
        "reward": reward_times,
    }


# ─── Noise Generation ───────────────────────────────────────────────────────

def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f noise via spectral shaping."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid div-by-zero
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n_samples)
    return pink / (np.std(pink) + 1e-12)


def _add_artifacts(signal: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Inject occasional large-amplitude motion artifacts."""
    n_ch, n_samp = signal.shape
    artifact_mask = rng.random(n_samp) < cfg.artifact_prob
    artifact_locs = np.where(artifact_mask)[0]
    for loc in artifact_locs:
        width = rng.integers(10, 60)
        amp = rng.uniform(3, 8) * cfg.noise_level
        lo = max(0, loc - width // 2)
        hi = min(n_samp, loc + width // 2)
        artifact = amp * np.sin(np.linspace(0, np.pi, hi - lo))
        for ch in range(n_ch):
            signal[ch, lo:hi] += artifact * (0.5 + rng.random())
    return signal


# ─── Main Generator ─────────────────────────────────────────────────────────

def generate(cfg: SimConfig = None, seed: int = 42) -> SimulatedData:
    """Generate complete simulated dataset."""
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.default_rng(seed)
    n_samples = int(cfg.duration_s * cfg.fs)

    # 1) Background noise (pink + white mix)
    signal = np.zeros((cfg.n_channels, n_samples))
    for ch in range(cfg.n_channels):
        pink = _pink_noise(n_samples, rng)
        white = rng.standard_normal(n_samples)
        signal[ch] = cfg.noise_level * (0.7 * pink + 0.3 * white)

    # 2) Behavioural events
    events = _generate_events(cfg, rng)

    # 3) Waveform templates
    templates = _make_waveform_templates(cfg, rng)

    # 4) Spike trains
    spike_trains = _generate_spike_trains(cfg, events, rng)

    # 5) Inject spikes into signal
    n_wf = templates[0].shape[1]
    for unit_id, indices in spike_trains.items():
        wf = templates[unit_id]
        for idx in indices:
            # Add amplitude jitter per spike
            jitter = 0.85 + 0.3 * rng.random()
            lo = idx
            hi = idx + n_wf
            if hi <= n_samples:
                signal[:, lo:hi] += wf * jitter

    # 6) Motion artifacts
    signal = _add_artifacts(signal, cfg, rng)

    return SimulatedData(
        raw_signal=signal,
        spike_trains=spike_trains,
        waveform_templates=templates,
        events=events,
        config=cfg,
    )


if __name__ == "__main__":
    data = generate()
    total_spikes = sum(len(v) for v in data.spike_trains.values())
    print(f"Generated {data.config.duration_s}s recording | "
          f"{data.config.n_channels} channels | "
          f"{data.config.n_units} units | "
          f"{total_spikes} total spikes | "
          f"{len(data.events['stimulus'])} trials")
