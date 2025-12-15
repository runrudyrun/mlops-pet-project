"""Unit tests for the configuration loader utility."""

import os
from pathlib import Path

import pytest
import yaml

from src.config import (
    DEFAULTS,
    deep_merge,
    get_config_value,
    load_config,
    process_config_value,
    substitute_env_vars,
)


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""

    def test_substitute_single_env_var(self, monkeypatch):
        """Test substitution of a single environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = substitute_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_substitute_env_var_in_string(self, monkeypatch):
        """Test substitution of env var embedded in a string."""
        monkeypatch.setenv("HOST", "localhost")
        result = substitute_env_vars("http://${HOST}:8000")
        assert result == "http://localhost:8000"

    def test_substitute_multiple_env_vars(self, monkeypatch):
        """Test substitution of multiple environment variables."""
        monkeypatch.setenv("USER", "admin")
        monkeypatch.setenv("PASS", "secret")
        result = substitute_env_vars("${USER}:${PASS}")
        assert result == "admin:secret"

    def test_missing_env_var_returns_empty(self):
        """Test that missing env vars are replaced with empty string."""
        # Ensure the var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)
        result = substitute_env_vars("${NONEXISTENT_VAR}")
        assert result == ""

    def test_no_env_vars_unchanged(self):
        """Test that strings without env vars are unchanged."""
        result = substitute_env_vars("plain string")
        assert result == "plain string"


class TestProcessConfigValue:
    """Tests for recursive config value processing."""

    def test_process_string_with_env_var(self, monkeypatch):
        """Test processing a string with env var."""
        monkeypatch.setenv("DB_HOST", "mydb.example.com")
        result = process_config_value("${DB_HOST}")
        assert result == "mydb.example.com"

    def test_process_dict_with_env_vars(self, monkeypatch):
        """Test processing a dict containing env vars."""
        monkeypatch.setenv("API_KEY", "secret123")
        config = {"api": {"key": "${API_KEY}", "timeout": 30}}
        result = process_config_value(config)
        assert result == {"api": {"key": "secret123", "timeout": 30}}

    def test_process_list_with_env_vars(self, monkeypatch):
        """Test processing a list containing env vars."""
        monkeypatch.setenv("HOST1", "server1")
        monkeypatch.setenv("HOST2", "server2")
        config = ["${HOST1}", "${HOST2}", "static"]
        result = process_config_value(config)
        assert result == ["server1", "server2", "static"]

    def test_process_nested_structure(self, monkeypatch):
        """Test processing deeply nested structures."""
        monkeypatch.setenv("NESTED_VAL", "found")
        config = {"level1": {"level2": [{"level3": "${NESTED_VAL}"}]}}
        result = process_config_value(config)
        assert result["level1"]["level2"][0]["level3"] == "found"

    def test_process_non_string_values(self):
        """Test that non-string values are preserved."""
        config = {"int": 42, "float": 3.14, "bool": True, "none": None}
        result = process_config_value(config)
        assert result == config


class TestDeepMerge:
    """Tests for deep dictionary merging."""

    def test_merge_flat_dicts(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_override_replaces_list(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_base_unchanged(self):
        """Test that the base dict is not modified."""
        base = {"a": 1}
        override = {"a": 2}
        deep_merge(base, override)
        assert base == {"a": 1}


class TestLoadConfig:
    """Tests for loading configuration from YAML files."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_content = {
            "data": {"raw_path": "custom/path.csv", "test_size": 0.3}
        }
        config_file = tmp_path / "params.yaml"
        config_file.write_text(yaml.dump(config_content))

        result = load_config(config_file)

        assert result["data"]["raw_path"] == "custom/path.csv"
        assert result["data"]["test_size"] == 0.3
        # Check defaults are applied
        assert result["data"]["random_state"] == 42

    def test_load_config_with_env_vars(self, tmp_path, monkeypatch):
        """Test loading config with environment variable substitution."""
        monkeypatch.setenv("MLFLOW_URI", "http://mlflow.example.com")
        config_content = {"mlflow": {"tracking_uri": "${MLFLOW_URI}"}}
        config_file = tmp_path / "params.yaml"
        config_file.write_text(yaml.dump(config_content))

        result = load_config(config_file)

        assert result["mlflow"]["tracking_uri"] == "http://mlflow.example.com"

    def test_load_config_applies_defaults(self, tmp_path):
        """Test that defaults are applied for missing keys."""
        config_content = {"data": {"test_size": 0.25}}
        config_file = tmp_path / "params.yaml"
        config_file.write_text(yaml.dump(config_content))

        result = load_config(config_file)

        # Check that defaults are applied
        assert "models" in result
        assert "evaluation" in result
        assert "monitoring" in result
        assert result["evaluation"]["primary_metric"] == "f1_score"

    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/path/config.yaml")

    def test_load_empty_config_uses_defaults(self, tmp_path):
        """Test that empty config file uses all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_config(config_file)

        assert result == DEFAULTS

    def test_load_config_with_path_object(self, tmp_path):
        """Test loading config using Path object."""
        config_content = {"data": {"test_size": 0.15}}
        config_file = tmp_path / "params.yaml"
        config_file.write_text(yaml.dump(config_content))

        result = load_config(Path(config_file))

        assert result["data"]["test_size"] == 0.15


class TestGetConfigValue:
    """Tests for getting nested config values."""

    def test_get_top_level_value(self):
        """Test getting a top-level config value."""
        config = {"key": "value"}
        result = get_config_value(config, "key")
        assert result == "value"

    def test_get_nested_value(self):
        """Test getting a nested config value using dot notation."""
        config = {"data": {"test_size": 0.2, "random_state": 42}}
        result = get_config_value(config, "data.test_size")
        assert result == 0.2

    def test_get_deeply_nested_value(self):
        """Test getting a deeply nested value."""
        config = {"level1": {"level2": {"level3": "deep_value"}}}
        result = get_config_value(config, "level1.level2.level3")
        assert result == "deep_value"

    def test_get_missing_key_returns_default(self):
        """Test that missing key returns the default value."""
        config = {"existing": "value"}
        result = get_config_value(config, "missing", default="fallback")
        assert result == "fallback"

    def test_get_missing_nested_key_returns_default(self):
        """Test that missing nested key returns the default value."""
        config = {"data": {"test_size": 0.2}}
        result = get_config_value(config, "data.nonexistent", default=None)
        assert result is None

    def test_get_value_default_is_none(self):
        """Test that default is None when not specified."""
        config = {}
        result = get_config_value(config, "missing")
        assert result is None
