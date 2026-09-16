import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import yaml

from gpt.config.data_config import DataConfig
from gpt.config.inference_config import InferenceConfig
from gpt.config.model_config import ModelConfig
from gpt.config.train_config import TrainingConfig


def expand_env_vars(text: str) -> str:
    """Expand environment variables matching ${VAR_NAME} or $VAR_NAME."""
    pattern = re.compile(r"\$(\w+)|\${(\w+)}")

    def replacer(match: re.Match) -> str:
        var = match.group(1) or match.group(2)
        return os.environ.get(var, "")

    return pattern.sub(replacer, text)


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = expand_env_vars(f.read())

    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(content) or {}
    elif path.suffix == ".json":
        import json
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use .yaml, .yml, or .json")


def save_config_file(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration dictionary to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            import json
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format '{path.suffix}'.")


def load_full_experiment_config(
    path: Union[str, Path],
) -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
    """Load combined experiment configuration containing model, training, and data sections."""
    raw = load_config_file(path)
    model_cfg = ModelConfig.from_dict(raw.get("model", {}))
    train_cfg = TrainingConfig.from_dict(raw.get("training", {}))
    data_cfg = DataConfig.from_dict(raw.get("data", {}))
    return model_cfg, train_cfg, data_cfg
