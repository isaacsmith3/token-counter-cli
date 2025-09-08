"""CLI argument parsing and configuration for token counter."""

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class InputSource(Enum):
    """Source of input data."""

    STDIN = "stdin"
    FILE = "file"
    MESSAGES = "messages"


@dataclass
class CLIConfig:
    """Configuration parsed from CLI arguments."""

    models: List[str]
    input_source: InputSource
    input_path: Optional[Path]
    max_tokens: Optional[int]
    reserve: Optional[int]
    reserve_pct: float
    json_output: bool

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_reserve_values()
        self._validate_models()

    def _validate_reserve_values(self) -> None:
        """Validate reserve and reserve_pct values."""
        if self.reserve is not None and self.reserve < 0:
            raise ValueError("--reserve must be non-negative")

        if not (0.0 <= self.reserve_pct <= 1.0):
            raise ValueError("--reserve-pct must be in range [0.0, 1.0]")

    def _validate_models(self) -> None:
        """Validate model names."""
        valid_models = {"gpt-4o", "claude-3-5-sonnet"}
        for model in self.models:
            if model not in valid_models:
                raise ValueError(
                    f"Unknown model: {model}. Valid models: {', '.join(sorted(valid_models))}"
                )


class CLIArgumentParser:
    """Handles CLI argument parsing and validation."""

    def __init__(self) -> None:
        """Initialize the argument parser."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            prog="tc",
            description="Cross-model token counting command-line tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  echo "Hello world" | tc
  tc --file prompt.txt
  tc --messages conversation.json --json
  tc --model gpt-4o --reserve 1000
  tc --model claude-3-5-sonnet --reserve-pct 0.1
            """.strip(),
        )

        # Input source arguments (mutually exclusive)
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            "--file",
            type=Path,
            metavar="PATH",
            help="Read input from file instead of stdin",
        )
        input_group.add_argument(
            "--messages",
            type=Path,
            metavar="PATH",
            help="Read structured messages from JSON file",
        )

        # Model selection
        parser.add_argument(
            "--model",
            action="append",
            dest="models",
            metavar="MODEL",
            help="Model to count tokens for (can be repeated). Valid: gpt-4o, claude-3-5-sonnet",
        )

        # Output format
        parser.add_argument(
            "--json", action="store_true", help="Output results in JSON format"
        )

        # Budget configuration
        parser.add_argument(
            "--max-tokens", type=int, metavar="N", help="Override model context limit"
        )

        reserve_group = parser.add_mutually_exclusive_group()
        reserve_group.add_argument(
            "--reserve",
            type=int,
            metavar="N",
            help="Absolute number of tokens to reserve for output",
        )
        reserve_group.add_argument(
            "--reserve-pct",
            type=float,
            metavar="FLOAT",
            default=0.2,
            help="Percentage of context to reserve for output (default: 0.2)",
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> CLIConfig:
        """Parse command line arguments into CLIConfig.

        Args:
            args: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Parsed and validated configuration

        Raises:
            SystemExit: On argument parsing errors or validation failures
        """
        if args is None:
            args = sys.argv[1:]

        try:
            parsed_args = self.parser.parse_args(args)
            return self._build_config(parsed_args)
        except ValueError as e:
            self.parser.error(str(e))

    def _build_config(self, args: argparse.Namespace) -> CLIConfig:
        """Build CLIConfig from parsed arguments."""
        # Determine input source and path
        if args.messages:
            input_source = InputSource.MESSAGES
            input_path = args.messages
        elif args.file:
            input_source = InputSource.FILE
            input_path = args.file
        else:
            input_source = InputSource.STDIN
            input_path = None

        # Set default models if none specified
        models = args.models if args.models else ["gpt-4o", "claude-3-5-sonnet"]

        # Validate max_tokens
        if args.max_tokens is not None and args.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")

        return CLIConfig(
            models=models,
            input_source=input_source,
            input_path=input_path,
            max_tokens=args.max_tokens,
            reserve=args.reserve,
            reserve_pct=args.reserve_pct,
            json_output=args.json,
        )


def parse_cli_args(args: Optional[List[str]] = None) -> CLIConfig:
    """Parse CLI arguments and return configuration.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed and validated configuration

    Raises:
        SystemExit: On argument parsing errors or validation failures
    """
    parser = CLIArgumentParser()
    return parser.parse_args(args)
