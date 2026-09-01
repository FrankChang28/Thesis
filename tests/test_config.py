from subject_nirs.common.config import apply_dataclass_config
from subject_nirs.stage2.config import Config


def test_main_defaults_are_32d_residual() -> None:
    cfg = Config()
    assert cfg.structure_feature_dim == 32
    assert cfg.fusion_method == "residual"
    assert "relat_cons_32" in cfg.structure_feature_path


def test_film_and_16d_remain_configurable() -> None:
    cfg = Config()
    apply_dataclass_config(
        cfg,
        {
            "fusion_method": "film",
            "structure_feature_dim": 16,
            "structure_feature_path": "./artifacts/stage1/{test_id}/relat_cons_16/subject_features_all_op.npz",
        },
    )
    assert cfg.fusion_method == "film"
    assert cfg.structure_feature_dim == 16


def test_concatenation_is_configurable() -> None:
    cfg = Config()
    apply_dataclass_config(cfg, {"fusion_method": "concatenation"})
    cfg.validate()
    assert cfg.fusion_method == "concatenation"


def test_unknown_config_key_is_rejected() -> None:
    cfg = Config()
    try:
        apply_dataclass_config(cfg, {"typo": 1})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown YAML keys must fail fast")
