from pathlib import Path
from typing import Any, Protocol, TypeVar

import yaml
from dacite import Config, from_dict

ConfigT = TypeVar("ConfigT")


class ConfigurableModule(Protocol):
    def __init__(self, config: Any) -> None: ...
    def __call__(self, x: Any) -> Any: ...


_REGISTRY: dict[str, tuple[type[ConfigurableModule], type]] = {}


def register(name: str, module_class: type[ConfigurableModule], config_class: type) -> None:
    _REGISTRY[name] = (module_class, config_class)


def create_module(config_path: Path) -> ConfigurableModule:
    with open(config_path) as f:
        data = yaml.safe_load(f)

    module_type = data.pop("type")
    if module_type not in _REGISTRY:
        raise ValueError(
            f"Unknown module type: {module_type}. Available: {list(_REGISTRY.keys())}"
        )

    module_class, config_class = _REGISTRY[module_type]
    config = from_dict(
        data_class=config_class,
        data=data,
        config=Config(cast=[tuple]),
    )
    return module_class(config)
