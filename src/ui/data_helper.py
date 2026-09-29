"""Data helpers and sensor metadata mappings for Industrial UI."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "T24_combustor_inlet_temp": {
        "name": "Combustor Inlet Temp",
        "code": "T24",
        "unit": "°C",
        "description": "Combustor inlet temperature reflecting compressor discharge heating",
    },
    "T30_hpt_coolant_temp": {
        "name": "HPT Coolant Temp",
        "code": "T30",
        "unit": "°C",
        "description": "High-pressure turbine coolant temperature in the hot gas section",
    },
    "T50_lpt_outlet_temp": {
        "name": "LPT Outlet Temp (EGT)",
        "code": "T50",
        "unit": "°C",
        "description": "Low-pressure turbine exhaust gas temperature reflecting thermal margin",
    },
    "P30_hpc_outlet_pressure": {
        "name": "HPC Outlet Pressure",
        "code": "P30",
        "unit": "bar",
        "description": "High-pressure compressor discharge pressure",
    },
    "Nf_fan_speed_rpm": {
        "name": "Fan Rotational Speed",
        "code": "Nf",
        "unit": "RPM",
        "description": "Aerodynamically coupled low-pressure spool / fan rotational speed",
    },
    "Nc_core_speed_rpm": {
        "name": "Core Spool Speed",
        "code": "Nc",
        "unit": "RPM",
        "description": "Core gas generator shaft rotational speed tightly coupled to throttle load",
    },
    "vib_vibration_amplitude": {
        "name": "Bearing Vibration",
        "code": "VIB",
        "unit": "mm/s",
        "description": "Shaft bearing broadband vibration amplitude (RMS)",
    },
}


def get_sensor_info(col: str) -> Dict[str, str]:
    """Retrieve human-friendly metadata for a sensor column."""
    if col in SENSOR_METADATA:
        return SENSOR_METADATA[col]
    clean_name = " ".join(col.split("_")).title()
    return {
        "name": clean_name,
        "code": col.split("_")[0].upper(),
        "unit": "",
        "description": f"Telemetry stream for {clean_name}",
    }
