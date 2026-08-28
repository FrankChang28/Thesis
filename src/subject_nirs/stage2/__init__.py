"""NIRS LOSO 1D-CNN training package.

The package is split by responsibility so training code is easier to maintain:
configuration, cache/preprocessing, sampling, datasets, model, engine, and LOSO workflow.
"""

from .config import CFG, DEVICE, Config

__all__ = ["CFG", "DEVICE", "Config"]
