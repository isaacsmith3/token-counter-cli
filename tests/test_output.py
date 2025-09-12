"""Tests for output formatting functionality."""

import json
import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from token_counter_cli.budget import BudgetResult
from token_counter_cli.output import OutputFormatter, format_human_readable, format_json


class TestOutputFormatter:
    """Test cases for OutputFormatter class."""

    def test_format_human_readable_empty_results(self):
        """Test formatting empty results list."""
        formatter = OutputFormatter()
        result = formatter.format_human_readable([])
        assert result == ""

    def test_format_human_readable_single_result(self):
        """Test formatting single result."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=150,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=102250,
                pct_used=0.12,
                warning=None,
                error=None,
            )
        ]

        output = formatter.format_human_readable(results)
        lines = output.split("\n")

        # Check header
        assert "model" in lines[0]
        assert "input_tokens" in lines[0]
        assert "context_limit" in lines[0]
        assert "pct_used" in lines[0]
        assert "remaining_tokens" in lines[0]
        assert "warnings" in lines[0]

        # Check data row
        assert "gpt-4o" in lines[1]
        assert "150" in lines[1]
        assert "128000" in lines[1]
        assert "0.12%" in lines[1]
        assert "102250" in lines[1]

    def test_format_human_readable_multiple_results(self):
        """Test formatting multiple results."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=150,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=102250,
                pct_used=0.12,
                warning=None,
                error=None,
            ),
            BudgetResult(
                model="claude-3-5-sonnet",
                input_tokens=200,
                context_limit=200000,
                effective_limit=200000,
                reserve=40000,
                remaining_tokens=159800,
                pct_used=0.10,
                warning=None,
                error=None,
            ),
        ]

        output = formatter.format_human_readable(results)
        lines = output.split("\n")

        # Should have header + 2 data rows
        assert len(lines) == 3
        assert "gpt-4o" in lines[1]
        assert "claude-3-5-sonnet" in lines[2]

    def test_format_human_readable_with_warning(self):
        """Test formatting result with warning."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=102400,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=0,
                pct_used=0.80,
                warning="warning: near limit",
                error=None,
            )
        ]

        output = formatter.format_human_readable(results)
        assert "warning: near limit" in output

    def test_format_human_readable_with_error(self):
        """Test formatting result with error."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=130000,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=-27600,
                pct_used=1.02,
                warning=None,
                error="error: exceeds budget",
            )
        ]

        output = formatter.format_human_readable(results)
        assert "error: exceeds budget" in output

    def test_format_json_empty_results(self):
        """Test JSON formatting with empty results."""
        formatter = OutputFormatter()
        result = formatter.format_json([])
        parsed = json.loads(result)
        assert parsed == []

    def test_format_json_single_result(self):
        """Test JSON formatting with single result."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=150,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=102250,
                pct_used=0.12,
                warning=None,
                error=None,
            )
        ]

        output = formatter.format_json(results)
        parsed = json.loads(output)

        assert len(parsed) == 1
        result_json = parsed[0]

        # Check all required fields
        assert result_json["model"] == "gpt-4o"
        assert result_json["input_tokens"] == 150
        assert result_json["context_limit"] == 128000
        assert result_json["pct_used"] == 0.12
        assert result_json["remaining_tokens"] == 102250
        assert result_json["warning"] is None
        assert result_json["error"] is None

    def test_format_json_with_warning_and_error(self):
        """Test JSON formatting with warning and error results."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=102400,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=0,
                pct_used=0.80,
                warning="warning: near limit",
                error=None,
            ),
            BudgetResult(
                model="claude-3-5-sonnet",
                input_tokens=210000,
                context_limit=200000,
                effective_limit=200000,
                reserve=40000,
                remaining_tokens=-50000,
                pct_used=1.05,
                warning=None,
                error="error: exceeds budget",
            ),
        ]

        output = formatter.format_json(results)
        parsed = json.loads(output)

        assert len(parsed) == 2
        assert parsed[0]["warning"] == "warning: near limit"
        assert parsed[0]["error"] is None
        assert parsed[1]["warning"] is None
        assert parsed[1]["error"] == "error: exceeds budget"

    def test_format_json_schema_compliance(self):
        """Test that JSON output matches expected schema."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="test-model",
                input_tokens=100,
                context_limit=1000,
                effective_limit=1000,
                reserve=200,
                remaining_tokens=700,
                pct_used=0.10,
                warning="test warning",
                error="test error",
            )
        ]

        output = formatter.format_json(results)
        parsed = json.loads(output)

        # Verify schema
        result = parsed[0]
        expected_keys = {
            "model",
            "input_tokens",
            "context_limit",
            "pct_used",
            "remaining_tokens",
            "warning",
            "error",
        }
        assert set(result.keys()) == expected_keys

        # Verify types
        assert isinstance(result["model"], str)
        assert isinstance(result["input_tokens"], int)
        assert isinstance(result["context_limit"], int)
        assert isinstance(result["pct_used"], float)
        assert isinstance(result["remaining_tokens"], int)
        assert isinstance(result["warning"], str)
        assert isinstance(result["error"], str)


class TestColorHandling:
    """Test cases for color handling functionality."""

    def test_colors_disabled_by_no_color_env(self):
        """Test that NO_COLOR environment variable disables colors."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            formatter = OutputFormatter()
            assert not formatter._colors_enabled

    def test_colors_disabled_by_non_tty(self):
        """Test that colors are disabled when stdout is not a TTY."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            formatter = OutputFormatter()
            assert not formatter._colors_enabled

    def test_colors_enabled_by_default(self):
        """Test that colors are enabled by default when conditions are met."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                formatter = OutputFormatter()
                assert formatter._colors_enabled

    def test_colorize_with_colors_enabled(self):
        """Test colorization when colors are enabled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                formatter = OutputFormatter()

                red_text = formatter._colorize("error", "red")
                assert red_text == "\033[31merror\033[0m"

                yellow_text = formatter._colorize("warning", "yellow")
                assert yellow_text == "\033[33mwarning\033[0m"

    def test_colorize_with_colors_disabled(self):
        """Test colorization when colors are disabled."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            formatter = OutputFormatter()

            red_text = formatter._colorize("error", "red")
            assert red_text == "error"

            yellow_text = formatter._colorize("warning", "yellow")
            assert yellow_text == "warning"

    def test_colorize_unknown_color(self):
        """Test colorization with unknown color name."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                formatter = OutputFormatter()

                result = formatter._colorize("text", "unknown")
                assert result == "text"

    def test_human_readable_output_with_colors(self):
        """Test that human-readable output includes colors for warnings/errors."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                formatter = OutputFormatter()

                results = [
                    BudgetResult(
                        model="test1",
                        input_tokens=100,
                        context_limit=1000,
                        effective_limit=1000,
                        reserve=200,
                        remaining_tokens=700,
                        pct_used=0.80,
                        warning="warning: near limit",
                        error=None,
                    ),
                    BudgetResult(
                        model="test2",
                        input_tokens=200,
                        context_limit=1000,
                        effective_limit=1000,
                        reserve=200,
                        remaining_tokens=-400,
                        pct_used=1.05,
                        warning=None,
                        error="error: exceeds budget",
                    ),
                ]

                output = formatter.format_human_readable(results)

                # Should contain ANSI color codes
                assert "\033[33m" in output  # yellow for warning
                assert "\033[31m" in output  # red for error
                assert "\033[0m" in output  # reset code

    def test_human_readable_output_without_colors(self):
        """Test that human-readable output excludes colors when disabled."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            formatter = OutputFormatter()

            results = [
                BudgetResult(
                    model="test",
                    input_tokens=100,
                    context_limit=1000,
                    effective_limit=1000,
                    reserve=200,
                    remaining_tokens=-400,
                    pct_used=1.05,
                    warning=None,
                    error="error: exceeds budget",
                )
            ]

            output = formatter.format_human_readable(results)

            # Should not contain ANSI color codes
            assert "\033[" not in output


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_format_human_readable_function(self):
        """Test format_human_readable convenience function."""
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=150,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=102250,
                pct_used=0.12,
                warning=None,
                error=None,
            )
        ]

        output = format_human_readable(results)
        assert "gpt-4o" in output
        assert "150" in output

    def test_format_json_function(self):
        """Test format_json convenience function."""
        results = [
            BudgetResult(
                model="gpt-4o",
                input_tokens=150,
                context_limit=128000,
                effective_limit=128000,
                reserve=25600,
                remaining_tokens=102250,
                pct_used=0.12,
                warning=None,
                error=None,
            )
        ]

        output = format_json(results)
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["model"] == "gpt-4o"


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_percentage_formatting(self):
        """Test that percentages are formatted correctly with 2 decimal places."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="test",
                input_tokens=1,
                context_limit=3,
                effective_limit=3,
                reserve=0,
                remaining_tokens=2,
                pct_used=0.33,  # Already rounded by budget analyzer
                warning=None,
                error=None,
            )
        ]

        output = formatter.format_human_readable(results)
        assert "0.33%" in output

        json_output = formatter.format_json(results)
        parsed = json.loads(json_output)
        assert parsed[0]["pct_used"] == 0.33

    def test_long_model_names(self):
        """Test formatting with long model names."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="very-long-model-name-that-exceeds-normal-width",
                input_tokens=100,
                context_limit=1000,
                effective_limit=1000,
                reserve=200,
                remaining_tokens=700,
                pct_used=0.10,
                warning=None,
                error=None,
            )
        ]

        output = formatter.format_human_readable(results)
        lines = output.split("\n")

        # Should handle long names gracefully
        assert len(lines) == 2  # header + data
        assert "very-long-model-name-that-exceeds-normal-width" in lines[1]

    def test_negative_remaining_tokens(self):
        """Test formatting with negative remaining tokens."""
        formatter = OutputFormatter()
        results = [
            BudgetResult(
                model="test",
                input_tokens=1500,
                context_limit=1000,
                effective_limit=1000,
                reserve=200,
                remaining_tokens=-700,
                pct_used=1.50,
                warning=None,
                error="error: exceeds budget",
            )
        ]

        output = formatter.format_human_readable(results)
        assert "-700" in output

        json_output = formatter.format_json(results)
        parsed = json.loads(json_output)
        assert parsed[0]["remaining_tokens"] == -700
