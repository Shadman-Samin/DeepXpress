import os
from pathlib import Path
import pytest

from configs.config import Config


class TestConfig:
    def test_default_model_name(self):
        c = Config()
        assert c.model_name == "mobilefacenet_onnx"

    def test_default_num_classes(self):
        c = Config()
        assert c.num_classes == 7

    def test_default_batch_size(self):
        c = Config()
        assert c.batch_size == 64

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEPXPRESS_BATCH_SIZE", "128")
        c = Config()
        assert c.batch_size == 128

    def test_default_input_size(self):
        c = Config()
        assert c.input_size == (112, 112)
