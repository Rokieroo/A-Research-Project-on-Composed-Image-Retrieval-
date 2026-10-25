from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
REQUIRED_KEYS = {"name", "dataset", "split", "data_root", "model", "eval", "output_dir"}


def test_configs_exist():
    assert CONFIGS, "configs/ should contain at least one experiment"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_is_valid(path):
    cfg = yaml.safe_load(path.read_text())
    assert REQUIRED_KEYS <= cfg.keys(), f"missing keys: {REQUIRED_KEYS - cfg.keys()}"
    assert cfg["name"] == path.stem, "config name should match its file name"
    assert cfg["dataset"] in {"fashioniq", "cirr", "circo"}
    assert "backbone" in cfg["model"]
