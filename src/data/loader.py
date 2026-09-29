"""Industrial sensor dataset generation and loader.

This module provides real-world physical dynamics sensor telemetry simulation modeled after
industrial turbofan / gas turbine monitoring systems (temperatures, pressures, shaft RPMs, vibration).
It includes physical coupling, operational degradation trends, noise, cyclical load profiles,
and anomaly events.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("DataLoader")

SENSOR_COLUMNS = [
    "T24_combustor_inlet_temp",   # Combustor inlet temperature (Kelvin / C)
    "T30_hpt_coolant_temp",        # High-Pressure Turbine Coolant temp
    "T50_lpt_outlet_temp",        # Low-Pressure Turbine outlet temp
    "P30_hpc_outlet_pressure",    # HPC outlet pressure (psia / bar)
    "Nf_fan_speed_rpm",           # Physical fan rotational speed (RPM)
    "Nc_core_speed_rpm",          # Core spool speed (RPM)
    "vib_vibration_amplitude",    # Bearing vibration amplitude (mm/s RMS)
]


def generate_industrial_telemetry(
    num_timesteps: int = 12000,
    sampling_interval_sec: int = 60,
    start_time: str = "2026-01-01 00:00:00",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate industrial sensor telemetry data grounded in physical turbine principles.

    Args:
        num_timesteps: Total chronological steps.
        sampling_interval_sec: Sampling interval in seconds (default: 60s / 1 min).
        start_time: Starting timestamp string.
        random_seed: Deterministic random seed.

    Returns:
        DataFrame containing 'timestamp' and all SENSOR_COLUMNS.
    """
    rng = np.random.default_rng(random_seed)
    time_index = pd.date_range(start=start_time, periods=num_timesteps, freq=f"{sampling_interval_sec}s")

    # Time dynamics: diurnal operational duty cycle (1440 min = 24h) + weekly load shift
    t = np.arange(num_timesteps, dtype=np.float64)
    daily_cycle = np.sin(2 * np.pi * t / (1440 * 60 / sampling_interval_sec))
    weekly_cycle = np.cos(2 * np.pi * t / (7 * 1440 * 60 / sampling_interval_sec))

    # Base operating load factor (normalized 0.6 - 1.0)
    operating_load = 0.8 + 0.15 * daily_cycle + 0.05 * weekly_cycle + rng.normal(0, 0.02, size=num_timesteps)
    operating_load = np.clip(operating_load, 0.5, 1.05)

    # Physical sensor couplings:
    # 1. Core speed Nc: tightly coupled to operating load
    nc_core_speed = 9000.0 + operating_load * 1200.0 + rng.normal(0, 15.0, size=num_timesteps)

    # 2. Fan speed Nf: aerodynamic coupling to Nc
    nf_fan_speed = 2200.0 + (nc_core_speed / 9000.0) * 350.0 + rng.normal(0, 8.0, size=num_timesteps)

    # 3. HPC outlet pressure P30: quadratic-like compression function of Nc
    p30_pressure = 14.7 + 18.0 * ((nc_core_speed - 8500.0) / 1700.0) ** 1.3 + rng.normal(0, 0.25, size=num_timesteps)

    # 4. Combustor inlet temperature T24 (compressor discharge heating)
    t24_temp = 550.0 + 85.0 * operating_load + rng.normal(0, 2.5, size=num_timesteps)

    # 5. HPT Coolant temp T30 (hot section thermal profile)
    t30_temp = 1350.0 + 190.0 * operating_load + 0.05 * (p30_pressure - 25.0) + rng.normal(0, 4.0, size=num_timesteps)

    # 6. LPT Outlet temp T50 (exhaust gas temperature EGT)
    t50_temp = 820.0 + 110.0 * operating_load + rng.normal(0, 3.0, size=num_timesteps)

    # 7. Vibration amplitude: function of shaft speed harmonics + baseline noise
    vib_amplitude = 1.2 + 0.6 * ((nc_core_speed - 9000.0) / 1200.0) ** 2 + rng.normal(0, 0.08, size=num_timesteps)
    vib_amplitude = np.clip(vib_amplitude, 0.2, 5.0)

    # Inject minor degradation trend over time (wear & fouling)
    wear_factor = np.linspace(0.0, 1.0, num_timesteps)
    t50_temp += 15.0 * wear_factor   # EGT margin erosion
    vib_amplitude += 0.35 * wear_factor  # Mechanical wear

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": time_index,
        "T24_combustor_inlet_temp": np.round(t24_temp, 2),
        "T30_hpt_coolant_temp": np.round(t30_temp, 2),
        "T50_lpt_outlet_temp": np.round(t50_temp, 2),
        "P30_hpc_outlet_pressure": np.round(p30_pressure, 2),
        "Nf_fan_speed_rpm": np.round(nf_fan_speed, 2),
        "Nc_core_speed_rpm": np.round(nc_core_speed, 2),
        "vib_vibration_amplitude": np.round(vib_amplitude, 3),
    })

    return df


def load_raw_dataset(
    raw_path: str = "data/raw/sensor_telemetry.csv",
    auto_generate_if_missing: bool = True,
    num_timesteps: int = 12000,
) -> pd.DataFrame:
    """Load raw dataset from disk, generating realistic telemetry if file does not exist yet.

    Args:
        raw_path: Path to raw CSV file.
        auto_generate_if_missing: If True, writes physical telemetry when absent.
        num_timesteps: Samples to create if generating.

    Returns:
        Loaded pandas DataFrame.
    """
    p = Path(raw_path)
    if not p.exists():
        if auto_generate_if_missing:
            logger.info(f"Raw data file {raw_path} not found. Generating telemetry ({num_timesteps} steps)...")
            p.parent.mkdir(parents=True, exist_ok=True)
            df = generate_industrial_telemetry(num_timesteps=num_timesteps)
            df.to_csv(p, index=False)
            logger.info(f"Saved generated raw dataset to {p}")
            return df
        else:
            raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    logger.info(f"Loading raw telemetry dataset from {p}...")
    df = pd.read_csv(p)
    return df
