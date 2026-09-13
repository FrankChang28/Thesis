"""Canonical thesis-facing labels and units for Stage 2 outputs.

Model artifacts keep their native target keys and units.  These helpers are
used only when building human-readable tables and figures.
"""

from __future__ import annotations


BASELINE_DISPLAY = "DRS-only baseline"

TARGET_DISPLAY = {
    "hc": "tHb",
    "sto2": "StO₂",
}
TARGET_VALUE_UNIT = {
    "hc": "μM",
    "sto2": "%",
}
TARGET_ERROR_UNIT = {
    "hc": "μM",
    "sto2": "%",
}
TARGET_DISPLAY_SCALE = {
    "hc": 1.0,
    "sto2": 100.0,
}

FEATURE_DISPLAY = {
    "baseline": "—",
    "latent": "DTOF descriptor",
    "metadata": "Tissue-thickness feature",
    "subject feature": "Subject feature",
}
FUSION_DISPLAY = {
    "none": "—",
    "concatenation": "Concatenation",
    "film": "FiLM",
    "residual": "Gated residual",
}
STRATEGY_DISPLAY = {
    "scratch": "Scratch",
    "finetune": "Fine-tuning",
}

_TARGET_ALIASES = {
    "hc": "hc",
    "gm_hc": "hc",
    "thb": "hc",
    "sto2": "sto2",
    "sto₂": "sto2",
    "gm_sto2": "sto2",
}


def target_key(target: str) -> str:
    """Normalize internal and display target names to ``hc`` or ``sto2``."""
    normalized = str(target).strip().lower()
    try:
        return _TARGET_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown target label: {target!r}") from exc


def target_display(target: str) -> str:
    return TARGET_DISPLAY[target_key(target)]


def target_value_unit(target: str) -> str:
    return TARGET_VALUE_UNIT[target_key(target)]


def target_error_unit(target: str) -> str:
    return TARGET_ERROR_UNIT[target_key(target)]


def target_display_scale(target: str) -> float:
    """Return the native-to-thesis scale (StO₂ fraction becomes percent)."""
    return TARGET_DISPLAY_SCALE[target_key(target)]
