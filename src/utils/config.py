"""
Configuration Utilities
=======================

Loading, merging, and saving configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import copy


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If required fields are missing
    """
    required_sections = ["training", "policy_optimization", "rewards"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Check training config
    training_required = ["num_epochs", "batch_size", "learning_rate"]
    for field in training_required:
        if field not in config["training"]:
            raise ValueError(f"Missing required training field: {field}")

    # Check policy optimization config
    po_required = ["algorithm", "K", "K_prime"]
    for field in po_required:
        if field not in config["policy_optimization"]:
            raise ValueError(f"Missing required policy_optimization field: {field}")

    return True
