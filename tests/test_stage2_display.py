from __future__ import annotations

import pytest

from subject_nirs.stage2.display import (
    BASELINE_DISPLAY,
    FEATURE_DISPLAY,
    FUSION_DISPLAY,
    STRATEGY_DISPLAY,
    target_display,
    target_display_scale,
    target_error_unit,
    target_value_unit,
)


def test_canonical_labels() -> None:
    assert BASELINE_DISPLAY == "DRS-only baseline"
    assert FEATURE_DISPLAY["latent"] == "DTOF descriptor"
    assert FEATURE_DISPLAY["metadata"] == "Tissue-thickness feature"
    assert FUSION_DISPLAY["residual"] == "Gated residual"
    assert STRATEGY_DISPLAY["finetune"] == "Fine-tuning"


def test_target_display_units_and_scale() -> None:
    assert target_display("HC") == "tHb"
    assert target_display("StO2") == "StO₂"
    assert target_value_unit("sto2") == "%"
    assert target_error_unit("sto2") == "%"
    assert target_display_scale("sto2") == 100.0
    assert target_display_scale("hc") == 1.0


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(ValueError):
        target_display("unknown")
