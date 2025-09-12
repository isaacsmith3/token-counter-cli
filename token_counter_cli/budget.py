"""Budget analysis functionality for token counting."""

from dataclasses import dataclass
from typing import Optional

from .cli import CLIConfig
from .counting import CountingResult
from .models import ModelDefinition


@dataclass
class BudgetResult:
    """Result of budget analysis for a model."""

    model: str
    input_tokens: int
    context_limit: int
    effective_limit: int
    reserve: int
    remaining_tokens: int
    pct_used: float  # rounded to 2 decimals
    warning: Optional[str] = None
    error: Optional[str] = None
    is_approximate: bool = False


class BudgetAnalyzer:
    """Handles budget analysis calculations and threshold checks."""

    def analyze_budget(
        self,
        counting_result: CountingResult,
        model: ModelDefinition,
        config: CLIConfig,
    ) -> BudgetResult:
        """Perform budget analysis with thresholds.

        Args:
            counting_result: Result from token counting
            model: Model definition with context limits
            config: CLI configuration with reserve settings

        Returns:
            BudgetResult with budget analysis and threshold warnings/errors
        """
        # If counting failed, return error result
        if counting_result.error:
            return BudgetResult(
                model=model.name,
                input_tokens=0,
                context_limit=model.context_limit,
                effective_limit=model.context_limit,
                reserve=0,
                remaining_tokens=0,
                pct_used=0.0,
                error=counting_result.error,
                is_approximate=counting_result.is_approximate,
            )

        # Calculate effective limit
        effective_limit = self._calculate_effective_limit(model, config)

        # Calculate reserve
        reserve = self._calculate_reserve(effective_limit, config)

        # Calculate remaining tokens
        remaining_tokens = effective_limit - reserve - counting_result.input_tokens

        # Calculate percentage used (rounded to 2 decimals)
        if effective_limit == 0:
            # Handle zero division case
            pct_used = (
                0.0 if counting_result.input_tokens == 0 else 999.99
            )  # Cap at a large but finite value
        else:
            pct_used = round(counting_result.input_tokens / effective_limit, 2)

        # Determine warning and error messages
        warning, error = self._check_thresholds(pct_used, remaining_tokens)

        return BudgetResult(
            model=model.name,
            input_tokens=counting_result.input_tokens,
            context_limit=model.context_limit,
            effective_limit=effective_limit,
            reserve=reserve,
            remaining_tokens=remaining_tokens,
            pct_used=pct_used,
            warning=warning,
            error=error,
            is_approximate=counting_result.is_approximate,
        )

    def _calculate_effective_limit(
        self, model: ModelDefinition, config: CLIConfig
    ) -> int:
        """Calculate effective limit based on context limit and max_tokens.

        Args:
            model: Model definition with context_limit
            config: CLI configuration with optional max_tokens

        Returns:
            Effective limit (min of context_limit and max_tokens if provided)
        """
        if config.max_tokens is not None:
            return min(model.context_limit, config.max_tokens)
        return model.context_limit

    def _calculate_reserve(self, effective_limit: int, config: CLIConfig) -> int:
        """Calculate reserve tokens based on configuration.

        Args:
            effective_limit: Effective context limit
            config: CLI configuration with reserve settings

        Returns:
            Number of tokens to reserve
        """
        if config.reserve is not None:
            # Use absolute reserve value
            return config.reserve
        else:
            # Use percentage-based reserve
            return int(effective_limit * config.reserve_pct)

    def _check_thresholds(
        self, pct_used: float, remaining_tokens: int
    ) -> tuple[Optional[str], Optional[str]]:
        """Check warning and error thresholds.

        Args:
            pct_used: Percentage of effective limit used
            remaining_tokens: Number of tokens remaining after reserve

        Returns:
            Tuple of (warning_message, error_message) where either can be None
        """
        warning = None
        error = None

        # Check error threshold first (higher priority)
        # Handle large percentage case (when effective_limit is 0 but input_tokens > 0)
        if pct_used >= 0.95 or remaining_tokens < 0:
            error = "error: exceeds budget"
        # Check warning threshold only if no error
        elif pct_used >= 0.80:
            warning = "warning: near limit"

        return warning, error


def analyze_budget(
    counting_result: CountingResult, model: ModelDefinition, config: CLIConfig
) -> BudgetResult:
    """Perform budget analysis with thresholds.

    This is a convenience function that creates a BudgetAnalyzer instance
    and calls the analyze_budget method.

    Args:
        counting_result: Result from token counting
        model: Model definition with context limits
        config: CLI configuration with reserve settings

    Returns:
        BudgetResult with budget analysis and threshold warnings/errors
    """
    analyzer = BudgetAnalyzer()
    return analyzer.analyze_budget(counting_result, model, config)
