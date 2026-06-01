from pathlib import Path
from typing import Any
import os

import yaml


class Config:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config: dict[str, Any] = {}
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.yaml"
        self._load(path=Path(config_path))
        self._override_from_env()

    def _load(self, path: Path) -> None:
        if path.exists():
            with open(path) as f:
                self._config = yaml.safe_load(f)

    def _override_from_env(self) -> None:
        env_mapping = {
            "DEEPXPRESS_MODEL_NAME": ("model", "name"),
            "DEEPXPRESS_BATCH_SIZE": ("training", "batch_size"),
            "DEEPXPRESS_EPOCHS": ("training", "epochs"),
            "DEEPXPRESS_LR": ("training", "learning_rate"),
            "DEEPXPRESS_DATA_DIR": ("data", "data_dir"),
        }
        for env_var, keys in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                section = self._config
                for key in keys[:-1]:
                    section = section.setdefault(key, {})
                section[keys[-1]] = self._cast(value)

    @staticmethod
    def _cast(value: str) -> int | float | bool | str:
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def get(self, *keys: str, default: Any = None) -> Any:
        section = self._config
        for key in keys:
            if isinstance(section, dict):
                section = section.get(key)
                if section is None:
                    return default
            else:
                return default
        return section

    @property
    def model_name(self) -> str:
        return str(self.get("model", "name", default="mobilefacenet_onnx"))

    @property
    def input_size(self) -> tuple[int, int]:
        size = self.get("model", "input_size", default=[112, 112])
        return tuple(size)

    @property
    def num_classes(self) -> int:
        return int(self.get("model", "num_classes", default=7))

    @property
    def batch_size(self) -> int:
        return int(self.get("training", "batch_size", default=64))

    @property
    def epochs(self) -> int:
        return int(self.get("training", "epochs", default=100))

    @property
    def learning_rate(self) -> float:
        return float(self.get("training", "learning_rate", default=0.001))


config = Config()
