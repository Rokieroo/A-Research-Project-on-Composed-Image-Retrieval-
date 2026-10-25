from pathlib import Path

import pytest
import yaml

from src.config import Config, ConfigError, config_from_dict, load_config, set_seed

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_exist():
    assert CONFIGS, "configs/ should contain at least one experiment"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.name == path.stem, "config name should match its file name"


def _minimal(**extra):
    return {"name": "x", "dataset": "cirr", **extra}


def test_defaults_are_filled_in():
    cfg = config_from_dict(_minimal())
    assert cfg.device == "auto"
    assert cfg.model.backbone == "ViT-B-32"
    assert cfg.eval.recall_ks == [1, 5, 10, 50]
    assert cfg.train is None
    assert cfg.dataset_dir == Path("data/cirr")


def test_overrides_apply(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(yaml.safe_dump(_minimal()))
    assert load_config(path, {"device": "cpu"}).device == "cpu"


def test_train_section_is_parsed():
    cfg = config_from_dict(_minimal(train={"epochs": 2, "lr": 1e-4}))
    assert cfg.train.epochs == 2
    assert cfg.train.batch_size == 32


@pytest.mark.parametrize(
    "raw",
    [
        {"dataset": "cirr"},  # missing name
        _minimal(dataset="coco"),
        _minimal(split="dev"),
        _minimal(device="tpu"),
        _minimal(typo_key=1),
        _minimal(model={"fusion": "concat"}),
        _minimal(model={"backbon": "ViT-B-32"}),
        _minimal(eval={"recall_ks": [0, 5]}),
        _minimal(eval={"subset_ks": [1, 6]}),
        _minimal(categories=["shoes"]),
        _minimal(train={"lr": -1}),
    ],
)
def test_invalid_configs_are_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_missing_file_is_rejected():
    with pytest.raises(ConfigError):
        load_config("configs/does_not_exist.yaml")


def test_set_seed_is_reproducible():
    import numpy as np

    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    assert (np.random.rand(3) == a).all()
