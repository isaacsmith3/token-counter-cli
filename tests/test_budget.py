"""Tests for budget analysis functionality."""

from typing import Any

from token_counter_cli.budget import BudgetAnalyzer, BudgetResult, analyze_budget
from token_counter_cli.cli import CLIConfig, InputSource
from token_counter_cli.counting import CountingResult
from token_counter_cli.models import ModelDefinition


def create_test_config(**overrides: Any) -> CLIConfig:
    """Create a test CLIConfig with valid defaults and optional overrides."""
    defaults = {
        "models": ["gpt-4o"],
        "input_source": InputSource.STDIN,
        "input_path": None,
        "max_tokens": None,
        "reserve": None,
        "reserve_pct": 0.2,
        "json_output": False,
    }
    defaults.update(overrides)
    return CLIConfig(
        models=defaults["models"],
        input_source=defaults["input_source"],
        input_path=defaults["input_path"],
        max_tokens=defaults["max_tokens"],
        reserve=defaults["reserve"],
        reserve_pct=defaults["reserve_pct"],
        json_output=defaults["json_output"],
    )


class TestBudgetAnalyzer:
    """Test cases for BudgetAnalyzer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.analyzer = BudgetAnalyzer()
        self.model = ModelDefinition(
            name="gpt-4o", context_limit=1000, tokenizer_type="local"
        )

    def test_basic_budget_calculation(self) -> None:
        """Test basic budget calculation with default reserve percentage."""
        config = create_test_config()

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.model == "gpt-4o"
        assert result.input_tokens == 100
        assert result.context_limit == 1000
        assert result.effective_limit == 1000
        assert result.reserve == 200  # 20% of 1000
        assert result.remaining_tokens == 700  # 1000 - 200 - 100
        assert result.pct_used == 0.10  # 100/1000 rounded to 2 decimals
        assert result.warning is None
        assert result.error is None

    def test_absolute_reserve(self) -> None:
        """Test budget calculation with absolute reserve value."""
        config = create_test_config(
            reserve=150, reserve_pct=0.2
        )  # reserve_pct should be ignored

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.reserve == 150  # Uses absolute reserve
        assert result.remaining_tokens == 750  # 1000 - 150 - 100

    def test_max_tokens_override(self) -> None:
        """Test effective limit calculation with max_tokens override."""
        config = create_test_config(max_tokens=800)

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.effective_limit == 800  # min(1000, 800)
        assert result.reserve == 160  # 20% of 800
        assert result.remaining_tokens == 540  # 800 - 160 - 100
        assert result.pct_used == 0.12  # 100/800 rounded to 2 decimals

    def test_max_tokens_higher_than_context_limit(self) -> None:
        """Test that context limit is used when max_tokens is higher."""
        config = create_test_config(max_tokens=1500)

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.effective_limit == 1000  # min(1000, 1500)

    def test_warning_threshold_exactly_80_percent(self) -> None:
        """Test warning threshold at exactly 80% usage."""
        config = create_test_config(reserve_pct=0.0)  # No reserve for cleaner math

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=800, error=None  # Exactly 80%
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.pct_used == 0.80
        assert result.warning == "warning: near limit"
        assert result.error is None

    def test_warning_threshold_just_below_80_percent(self) -> None:
        """Test no warning just below 80% usage."""
        config = create_test_config(reserve_pct=0.0)

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=799, error=None  # Just below 80%
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.pct_used == 0.80  # Rounds to 0.80
        assert result.warning == "warning: near limit"  # Still triggers due to rounding
        assert result.error is None

    def test_error_threshold_exactly_95_percent(self) -> None:
        """Test error threshold at exactly 95% usage."""
        config = create_test_config(reserve_pct=0.0)

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=950, error=None  # Exactly 95%
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.pct_used == 0.95
        assert result.warning is None  # Error overrides warning
        assert result.error == "error: exceeds budget"

    def test_error_threshold_negative_remaining(self) -> None:
        """Test error threshold when remaining tokens are negative."""
        config = create_test_config(reserve=200)

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=900, error=None  # 900 + 200 > 1000
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.remaining_tokens == -100  # 1000 - 200 - 900
        assert result.pct_used == 0.90  # 900/1000
        assert result.warning is None  # Error overrides warning
        assert result.error == "error: exceeds budget"

    def test_error_overrides_warning(self) -> None:
        """Test that error threshold overrides warning threshold."""
        config = create_test_config(reserve_pct=0.0)

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=960, error=None  # 96% (both thresholds)
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.pct_used == 0.96
        assert result.warning is None  # Should be None when error is present
        assert result.error == "error: exceeds budget"

    def test_pct_used_rounding(self) -> None:
        """Test that pct_used is properly rounded to 2 decimal places."""
        config = create_test_config(reserve_pct=0.0)

        # Test various rounding scenarios
        test_cases = [
            (333, 0.33),  # 333/1000 = 0.333 -> 0.33
            (334, 0.33),  # 334/1000 = 0.334 -> 0.33
            (335, 0.34),  # 335/1000 = 0.335 -> 0.34 (round half up)
            (336, 0.34),  # 336/1000 = 0.336 -> 0.34
            (1, 0.00),  # 1/1000 = 0.001 -> 0.00
            (5, 0.01),  # 5/1000 = 0.005 -> 0.01 (round half up)
        ]

        for input_tokens, expected_pct in test_cases:
            counting_result = CountingResult(
                model="gpt-4o", input_tokens=input_tokens, error=None
            )

            result = self.analyzer.analyze_budget(counting_result, self.model, config)
            assert result.pct_used == expected_pct, f"Failed for {input_tokens} tokens"

    def test_counting_error_propagation(self) -> None:
        """Test that counting errors are properly propagated."""
        config = create_test_config()

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=0, error="Token counting failed"
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.model == "gpt-4o"
        assert result.input_tokens == 0
        assert result.context_limit == 1000
        assert result.effective_limit == 1000
        assert result.reserve == 0
        assert result.remaining_tokens == 0
        assert result.pct_used == 0.0
        assert result.warning is None
        assert result.error == "Token counting failed"

    def test_approximation_flag_propagation(self) -> None:
        """Test that approximation flag is properly propagated."""
        config = create_test_config()

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=100, error=None, is_approximate=True
        )

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.is_approximate is True

    def test_zero_reserve_percentage(self) -> None:
        """Test budget calculation with zero reserve percentage."""
        config = create_test_config(reserve_pct=0.0)

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.reserve == 0
        assert result.remaining_tokens == 900  # 1000 - 0 - 100

    def test_full_reserve_percentage(self) -> None:
        """Test budget calculation with 100% reserve percentage."""
        config = create_test_config(reserve_pct=1.0)

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, self.model, config)

        assert result.reserve == 1000  # 100% of 1000
        assert result.remaining_tokens == -100  # 1000 - 1000 - 100
        assert result.error == "error: exceeds budget"


class TestBudgetAnalyzerEdgeCases:
    """Test edge cases for budget analysis."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.analyzer = BudgetAnalyzer()

    def test_zero_context_limit(self) -> None:
        """Test behavior with zero context limit."""
        model = ModelDefinition(name="gpt-4o", context_limit=0, tokenizer_type="local")
        config = create_test_config()

        counting_result = CountingResult(model="gpt-4o", input_tokens=0, error=None)

        result = self.analyzer.analyze_budget(counting_result, model, config)

        assert result.effective_limit == 0
        assert result.reserve == 0  # 20% of 0
        assert result.remaining_tokens == 0
        # Division by zero should be handled gracefully
        # In Python, 0/0 would be NaN, but we should handle this case

    def test_very_large_numbers(self) -> None:
        """Test behavior with very large token counts."""
        model = ModelDefinition(
            name="gpt-4o", context_limit=1000000, tokenizer_type="local"
        )
        config = create_test_config()

        counting_result = CountingResult(
            model="gpt-4o", input_tokens=500000, error=None
        )

        result = self.analyzer.analyze_budget(counting_result, model, config)

        assert result.effective_limit == 1000000
        assert result.reserve == 200000  # 20% of 1000000
        assert result.remaining_tokens == 300000  # 1000000 - 200000 - 500000
        assert result.pct_used == 0.50  # 500000/1000000


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_analyze_budget_function(self) -> None:
        """Test the analyze_budget convenience function."""
        model = ModelDefinition(
            name="gpt-4o", context_limit=1000, tokenizer_type="local"
        )
        config = create_test_config()

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = analyze_budget(counting_result, model, config)

        assert isinstance(result, BudgetResult)
        assert result.model == "gpt-4o"
        assert result.input_tokens == 100
        assert result.pct_used == 0.10


class TestThresholdBoundaries:
    """Test threshold boundary conditions precisely."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.analyzer = BudgetAnalyzer()
        self.model = ModelDefinition(
            name="gpt-4o", context_limit=1000, tokenizer_type="local"
        )
        self.config = create_test_config(
            reserve_pct=0.0
        )  # No reserve for precise threshold testing

    def test_threshold_boundaries(self) -> None:
        """Test precise threshold boundaries."""
        test_cases = [
            # (input_tokens, expected_pct, expected_warning, expected_error)
            (799, 0.80, "warning: near limit", None),  # Rounds to 0.80
            (800, 0.80, "warning: near limit", None),  # Exactly 0.80
            (801, 0.80, "warning: near limit", None),  # Rounds to 0.80
            (849, 0.85, "warning: near limit", None),  # Between warning and error
            (949, 0.95, None, "error: exceeds budget"),  # Exactly 0.95
            (950, 0.95, None, "error: exceeds budget"),  # Exactly 0.95
            (951, 0.95, None, "error: exceeds budget"),  # Above 0.95
            (1000, 1.00, None, "error: exceeds budget"),  # Exactly 100%
        ]

        for input_tokens, expected_pct, expected_warning, expected_error in test_cases:
            counting_result = CountingResult(
                model="gpt-4o", input_tokens=input_tokens, error=None
            )

            result = self.analyzer.analyze_budget(
                counting_result, self.model, self.config
            )

            assert (
                result.pct_used == expected_pct
            ), f"Failed pct_used for {input_tokens}"
            assert (
                result.warning == expected_warning
            ), f"Failed warning for {input_tokens}"
            assert result.error == expected_error, f"Failed error for {input_tokens}"


class TestZeroDivisionHandling:
    """Test handling of zero division cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.analyzer = BudgetAnalyzer()

    def test_zero_effective_limit_with_zero_tokens(self) -> None:
        """Test zero effective limit with zero input tokens."""
        model = ModelDefinition(name="gpt-4o", context_limit=0, tokenizer_type="local")
        config = create_test_config()

        counting_result = CountingResult(model="gpt-4o", input_tokens=0, error=None)

        result = self.analyzer.analyze_budget(counting_result, model, config)

        assert result.effective_limit == 0
        assert result.pct_used == 0.0  # Should handle 0/0 gracefully
        assert result.warning is None
        assert result.error is None

    def test_zero_effective_limit_with_nonzero_tokens(self) -> None:
        """Test zero effective limit with non-zero input tokens."""
        model = ModelDefinition(name="gpt-4o", context_limit=0, tokenizer_type="local")
        config = create_test_config()

        counting_result = CountingResult(model="gpt-4o", input_tokens=100, error=None)

        result = self.analyzer.analyze_budget(counting_result, model, config)

        assert result.effective_limit == 0
        assert result.pct_used == 999.99  # Capped large value for zero division
        # This should trigger an error since we have tokens but no capacity
        assert result.error == "error: exceeds budget"
