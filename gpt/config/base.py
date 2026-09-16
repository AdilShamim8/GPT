import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

try:
    import yaml
except ImportError:
    yaml = None

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with serialization and validation support."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration instance from dictionary, filtering unknown keys."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML serialization. Run `pip install pyyaml`.")
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls: Type[T], yaml_str: str) -> T:
        """Deserialize configuration from YAML string."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML deserialization. Run `pip install pyyaml`.")
        data = yaml.safe_load(yaml_str) or {}
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate configuration parameters. Override in subclasses."""
        pass

    def __post_init__(self) -> None:
        self.validate()
