"""Model registry for token counter CLI."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelDefinition:
    """Definition of a supported model."""

    name: str
    context_limit: int
    tokenizer_type: str  # "local" or "provider"


class ModelRegistry:
    """Registry of hardcoded model definitions."""

    def __init__(self):
        """Initialize with hardcoded model definitions."""
        self._models: Dict[str, ModelDefinition] = {
            "gpt-4o": ModelDefinition(
                name="gpt-4o", context_limit=128000, tokenizer_type="local"
            ),
            "claude-3-5-sonnet": ModelDefinition(
                name="claude-3-5-sonnet",
                context_limit=200000,
                tokenizer_type="provider",
            ),
        }

    def get_model(self, name: str) -> ModelDefinition:
        """Get model definition by name.

        Args:
            name: Model name to look up

        Returns:
            ModelDefinition for the requested model

        Raises:
            KeyError: If model name is not found
        """
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}")
        return self._models[name]

    def get_available_models(self) -> list[str]:
        """Get list of available model names.

        Returns:
            List of available model names
        """
        return list(self._models.keys())
