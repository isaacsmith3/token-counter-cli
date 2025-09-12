"""Output formatting for token counter CLI."""

import json
import os
import sys
from typing import List

from .budget import BudgetResult


class OutputFormatter:
    """Handles output formatting for both human-readable and JSON formats."""

    def __init__(self):
        """Initialize the output formatter."""
        self._colors_enabled = self._should_enable_colors()

    def format_human_readable(self, results: List[BudgetResult]) -> str:
        """Generate space-separated table output.

        Args:
            results: List of budget analysis results

        Returns:
            Human-readable table as string
        """
        if not results:
            return ""

        # Define column headers and widths
        headers = [
            "model",
            "input_tokens",
            "context_limit",
            "pct_used",
            "remaining_tokens",
            "warnings",
        ]

        # Calculate column widths based on content
        rows = []
        for result in results:
            # Format percentage with 2 decimal places
            pct_str = f"{result.pct_used:.2f}%"

            # Combine warning and error messages
            warnings_str = ""
            if result.error:
                warnings_str = result.error
            elif result.warning:
                warnings_str = result.warning

            row = [
                result.model,
                str(result.input_tokens),
                str(result.context_limit),
                pct_str,
                str(result.remaining_tokens),
                warnings_str,
            ]
            rows.append(row)

        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        # Build output lines
        lines = []

        # Header line
        header_parts = []
        for i, header in enumerate(headers):
            header_parts.append(header.ljust(col_widths[i]))
        lines.append("  ".join(header_parts))

        # Data lines with color coding
        for i, row in enumerate(rows):
            result = results[i]
            row_parts = []

            for j, cell in enumerate(row):
                formatted_cell = cell.ljust(col_widths[j])

                # Apply color coding to warnings column
                if j == len(headers) - 1 and cell:  # warnings column
                    if result.error:
                        formatted_cell = self._colorize(formatted_cell, "red")
                    elif result.warning:
                        formatted_cell = self._colorize(formatted_cell, "yellow")

                row_parts.append(formatted_cell)

            lines.append("  ".join(row_parts))

        return "\n".join(lines)

    def format_json(self, results: List[BudgetResult]) -> str:
        """Generate JSON array output.

        Args:
            results: List of budget analysis results

        Returns:
            JSON string with proper schema
        """
        json_results = []

        for result in results:
            json_result = {
                "model": result.model,
                "input_tokens": result.input_tokens,
                "context_limit": result.context_limit,
                "pct_used": result.pct_used,
                "remaining_tokens": result.remaining_tokens,
                "warning": result.warning,
                "error": result.error,
            }
            json_results.append(json_result)

        return json.dumps(json_results, indent=2)

    def _should_enable_colors(self) -> bool:
        """Determine if colors should be enabled.

        Returns:
            True if colors should be enabled, False otherwise
        """
        # Check NO_COLOR environment variable
        if os.environ.get("NO_COLOR"):
            return False

        # Check if stdout is a TTY
        if not sys.stdout.isatty():
            return False

        return True

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color codes to text if colors are enabled.

        Args:
            text: Text to colorize
            color: Color name ("red", "yellow")

        Returns:
            Colorized text or original text if colors disabled
        """
        if not self._colors_enabled:
            return text

        color_codes = {"red": "\033[31m", "yellow": "\033[33m", "reset": "\033[0m"}

        if color not in color_codes:
            return text

        return f"{color_codes[color]}{text}{color_codes['reset']}"


def format_human_readable(results: List[BudgetResult]) -> str:
    """Generate space-separated table output.

    This is a convenience function that creates an OutputFormatter instance
    and calls the format_human_readable method.

    Args:
        results: List of budget analysis results

    Returns:
        Human-readable table as string
    """
    formatter = OutputFormatter()
    return formatter.format_human_readable(results)


def format_json(results: List[BudgetResult]) -> str:
    """Generate JSON array output.

    This is a convenience function that creates an OutputFormatter instance
    and calls the format_json method.

    Args:
        results: List of budget analysis results

    Returns:
        JSON string with proper schema
    """
    formatter = OutputFormatter()
    return formatter.format_json(results)
