import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "07_配置参数" / "data_paths.yaml"


def load_data_paths() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = os.environ.get("GOLDQ_RAW_DIR")
    processed_dir = os.environ.get("GOLDQ_PROCESSED_DIR")

    cfg["raw_dir"] = str(Path(raw_dir) if raw_dir else REPO_ROOT / cfg["raw_dir"])
    cfg["processed_dir"] = str(Path(processed_dir) if processed_dir else REPO_ROOT / cfg["processed_dir"])
    return cfg
