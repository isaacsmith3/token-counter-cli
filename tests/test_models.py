"""Tests for model registry functionality."""

import pytest
from token_counter_cli.models import ModelDefinition, ModelRegistry


class TestModelDefinition:
    """Tests for ModelDefinition dataclass."""

    def test_model_definition_creation(self):
        """Test creating a ModelDefinition."""
        model = ModelDefinition(
            name="test-model", context_limit=100000, tokenizer_type="local"
        )

        assert model.name == "test-model"
        assert model.context_limit == 100000
        assert model.tokenizer_type == "local"

    def test_model_definition_equality(self):
        """Test ModelDefinition equality comparison."""
        model1 = ModelDefinition("test", 100000, "local")
        model2 = ModelDefinition("test", 100000, "local")
        model3 = ModelDefinition("test", 200000, "local")

        assert model1 == model2
        assert model1 != model3


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_registry_initialization(self):
        """Test that registry initializes with hardcoded models."""
        registry = ModelRegistry()
        available_models = registry.get_available_models()

        assert "gpt-4o" in available_models
        assert "claude-3-5-sonnet" in available_models
        assert len(available_models) == 2

    def test_get_gpt4o_model(self):
        """Test getting gpt-4o model definition."""
        registry = ModelRegistry()
        model = registry.get_model("gpt-4o")

        assert model.name == "gpt-4o"
        assert model.context_limit == 128000
        assert model.tokenizer_type == "local"

    def test_get_claude_model(self):
        """Test getting claude-3-5-sonnet model definition."""
        registry = ModelRegistry()
        model = registry.get_model("claude-3-5-sonnet")

        assert model.name == "claude-3-5-sonnet"
        assert model.context_limit == 200000
        assert model.tokenizer_type == "provider"

    def test_get_unknown_model_raises_keyerror(self):
        """Test that getting unknown model raises KeyError."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="Unknown model: unknown-model"):
            registry.get_model("unknown-model")

    def test_get_available_models_returns_list(self):
        """Test that get_available_models returns correct list."""
        registry = ModelRegistry()
        models = registry.get_available_models()

        assert isinstance(models, list)
        assert set(models) == {"gpt-4o", "claude-3-5-sonnet"}

    def test_model_definitions_match_requirements(self):
        """Test that hardcoded models match requirements specifications."""
        registry = ModelRegistry()

        # Test gpt-4o matches requirements
        gpt4o = registry.get_model("gpt-4o")
        assert gpt4o.context_limit == 128000  # From requirements
        assert gpt4o.tokenizer_type == "local"  # Uses tiktoken locally

        # Test claude-3-5-sonnet matches requirements
        claude = registry.get_model("claude-3-5-sonnet")
        assert claude.context_limit == 200000  # From requirements
        assert claude.tokenizer_type == "provider"  # Uses Anthropic API
