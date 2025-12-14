"""Configuration loader utility for the MLOps pipeline.

This module provides functionality to load YAML configuration files
with environment variable substitution and default value handling.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


# Default configuration values
DEFAULTS = {
    "data": {
        "raw_path": "data/raw/wine_quality.csv",
        "processed_dir": "data/processed",
        "test_size": 0.2,
        "random_state": 42,
    },
    "models": [
        {
            "name": "random_forest",
            "enabled": True,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42,
            },
        }
    ],
    "evaluation": {
        "metrics_path": "reports/metrics.json",
        "primary_metric": "f1_score",
    },
    "monitoring": {
        "drift_threshold": 0.1,
        "report_path": "reports/drift_report.html",
    },
    "mlflow": {
        "tracking_uri": "",
        "experiment_name": "wine-quality-classification",
    },
}


def substitute_env_vars(value: str) -> str:
    """Substitute environment variables in a string.

    Supports ${VAR_NAME} syntax. If the environment variable is not set,
    the placeholder is replaced with an empty string.

    Args:
        value: String potentially containing ${VAR_NAME} placeholders.

    Returns:
        String with environment variables substituted.
    """
    pattern = r"\$\{([^}]+)\}"

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(pattern, replace, value)


def process_config_value(value: Any) -> Any:
    """Recursively process config values, substituting environment variables.

    Args:
        value: Configuration value (can be dict, list, string, or primitive).

    Returns:
        Processed value with environment variables substituted in strings.
    """
    if isinstance(value, dict):
        return {k: process_config_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [process_config_value(item) for item in value]
    elif isinstance(value, str):
        return substitute_env_vars(value)
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Values from override take precedence. Nested dicts are merged recursively.
    Lists are replaced entirely (not merged).

    Args:
        base: Base dictionary with default values.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path = "configs/params.yaml") -> dict:
    """Load configuration from YAML file with env var substitution and defaults.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary with environment variables substituted
        and default values applied for missing keys.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Apply defaults for missing top-level keys
    config = deep_merge(DEFAULTS, raw_config)

    # Process environment variable substitution
    config = process_config_value(config)

    return config


def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., "data.test_size").
        default: Default value if the key doesn't exist.

    Returns:
        The configuration value or default if not found.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
