"""Navigation system and metadata for Industrial AI Platform."""

from dataclasses import dataclass
from typing import List


@dataclass
class NavItem:
    key: str
    label: str
    icon_svg: str
    eyebrow: str
    title: str
    description: str


NAV_ITEMS: List[NavItem] = [
    NavItem(
        key="overview",
        label="Overview",
        icon_svg="dashboard",
        eyebrow="INDUSTRIAL AI CONTROL CENTER",
        title="Industrial Time-Series Intelligence",
        description="Real-time forecasting, anomaly detection and model evaluation for multivariate industrial telemetry.",
    ),
    NavItem(
        key="telemetry",
        label="Data & Telemetry",
        icon_svg="telemetry",
        eyebrow="TELEMETRY",
        title="Industrial Sensor Monitoring",
        description="Chronological multi-sensor dynamics, physical couplings, and parameter distributions.",
    ),
    NavItem(
        key="forecasting",
        label="Forecasting",
        icon_svg="forecasting",
        eyebrow="MULTI-STEP FORECASTING",
        title="Deep Learning Temporal Forecasting",
        description="Predict future sensor behavior using TensorFlow-based deep learning models.",
    ),
    NavItem(
        key="benchmarks",
        label="Model Benchmarks",
        icon_svg="benchmarks",
        eyebrow="MODEL BENCHMARKS",
        title="Empirical Architecture Comparison",
        description="Comparative evaluation across forecasting architectures on unscaled physical metrics.",
    ),
    NavItem(
        key="anomaly",
        label="Anomaly Detection",
        icon_svg="anomaly",
        eyebrow="ANOMALY DETECTION",
        title="Unsupervised Telemetry Fault Studio",
        description="Unsupervised detection of abnormal sensor behavior using LSTM Autoencoders and Isolation Forests.",
    ),
    NavItem(
        key="experiments",
        label="Experiments",
        icon_svg="experiments",
        eyebrow="EXPERIMENT LAB",
        title="ML Research Experiment Console",
        description="Systematic hyperparameter logs, training convergence, and architectural benchmark runs.",
    ),
    NavItem(
        key="research",
        label="Research",
        icon_svg="research",
        eyebrow="RESEARCH & THEORY",
        title="Theoretical Formulations & Findings",
        description="Formal self-attention derivations, empirical design rationale, and architectural benchmark insights.",
    ),
]


def get_nav_by_key(key: str) -> NavItem:
    for item in NAV_ITEMS:
        if item.key == key:
            return item
    return NAV_ITEMS[0]


def get_nav_by_label(label: str) -> NavItem:
    clean_label = label.strip()
    for item in NAV_ITEMS:
        if item.label.lower() in clean_label.lower() or item.key.lower() in clean_label.lower():
            return item
    return NAV_ITEMS[0]
