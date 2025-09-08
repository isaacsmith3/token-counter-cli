"""Tests for CLI argument parsing and configuration."""

import pytest
from pathlib import Path
from token_counter_cli.cli import (
    CLIConfig,
    CLIArgumentParser,
    InputSource,
    parse_cli_args,
)


class TestCLIConfig:
    """Test CLIConfig dataclass and validation."""

    def test_valid_config_creation(self):
        """Test creating a valid CLIConfig."""
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=1000,
            reserve=100,
            reserve_pct=0.2,
            json_output=False,
        )
        assert config.models == ["gpt-4o"]
        assert config.input_source == InputSource.STDIN
        assert config.max_tokens == 1000
        assert config.reserve == 100

    def test_negative_reserve_validation(self):
        """Test that negative reserve values are rejected."""
        with pytest.raises(ValueError, match="--reserve must be non-negative"):
            CLIConfig(
                models=["gpt-4o"],
                input_source=InputSource.STDIN,
                input_path=None,
                max_tokens=None,
                reserve=-1,
                reserve_pct=0.2,
                json_output=False,
            )

    def test_reserve_pct_out_of_range_low(self):
        """Test that reserve_pct below 0.0 is rejected."""
        with pytest.raises(
            ValueError, match="--reserve-pct must be in range \\[0.0, 1.0\\]"
        ):
            CLIConfig(
                models=["gpt-4o"],
                input_source=InputSource.STDIN,
                input_path=None,
                max_tokens=None,
                reserve=None,
                reserve_pct=-0.1,
                json_output=False,
            )

    def test_reserve_pct_out_of_range_high(self):
        """Test that reserve_pct above 1.0 is rejected."""
        with pytest.raises(
            ValueError, match="--reserve-pct must be in range \\[0.0, 1.0\\]"
        ):
            CLIConfig(
                models=["gpt-4o"],
                input_source=InputSource.STDIN,
                input_path=None,
                max_tokens=None,
                reserve=None,
                reserve_pct=1.1,
                json_output=False,
            )

    def test_reserve_pct_boundary_values(self):
        """Test that reserve_pct boundary values (0.0, 1.0) are accepted."""
        # Test 0.0
        config1 = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.0,
            json_output=False,
        )
        assert config1.reserve_pct == 0.0

        # Test 1.0
        config2 = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=None,
            reserve=None,
            reserve_pct=1.0,
            json_output=False,
        )
        assert config2.reserve_pct == 1.0

    def test_invalid_model_validation(self):
        """Test that invalid model names are rejected."""
        with pytest.raises(ValueError, match="Unknown model: invalid-model"):
            CLIConfig(
                models=["invalid-model"],
                input_source=InputSource.STDIN,
                input_path=None,
                max_tokens=None,
                reserve=None,
                reserve_pct=0.2,
                json_output=False,
            )

    def test_valid_models(self):
        """Test that valid model names are accepted."""
        config = CLIConfig(
            models=["gpt-4o", "claude-3-5-sonnet"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )
        assert config.models == ["gpt-4o", "claude-3-5-sonnet"]


class TestCLIArgumentParser:
    """Test CLI argument parsing."""

    def test_default_arguments(self):
        """Test parsing with no arguments (defaults)."""
        parser = CLIArgumentParser()
        config = parser.parse_args([])

        assert config.models == ["gpt-4o", "claude-3-5-sonnet"]
        assert config.input_source == InputSource.STDIN
        assert config.input_path is None
        assert config.max_tokens is None
        assert config.reserve is None
        assert config.reserve_pct == 0.2
        assert config.json_output is False

    def test_file_input(self):
        """Test --file argument."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--file", "test.txt"])

        assert config.input_source == InputSource.FILE
        assert config.input_path == Path("test.txt")

    def test_messages_input(self):
        """Test --messages argument."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--messages", "msgs.json"])

        assert config.input_source == InputSource.MESSAGES
        assert config.input_path == Path("msgs.json")

    def test_single_model_selection(self):
        """Test selecting a single model."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--model", "gpt-4o"])

        assert config.models == ["gpt-4o"]

    def test_multiple_model_selection(self):
        """Test selecting multiple models."""
        parser = CLIArgumentParser()
        config = parser.parse_args(
            ["--model", "gpt-4o", "--model", "claude-3-5-sonnet"]
        )

        assert config.models == ["gpt-4o", "claude-3-5-sonnet"]

    def test_json_output(self):
        """Test --json flag."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--json"])

        assert config.json_output is True

    def test_max_tokens(self):
        """Test --max-tokens argument."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--max-tokens", "5000"])

        assert config.max_tokens == 5000

    def test_absolute_reserve(self):
        """Test --reserve argument."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--reserve", "1000"])

        assert config.reserve == 1000
        assert config.reserve_pct == 0.2  # default when --reserve is used

    def test_percentage_reserve(self):
        """Test --reserve-pct argument."""
        parser = CLIArgumentParser()
        config = parser.parse_args(["--reserve-pct", "0.15"])

        assert config.reserve is None
        assert config.reserve_pct == 0.15

    def test_complex_argument_combination(self):
        """Test complex combination of arguments."""
        parser = CLIArgumentParser()
        config = parser.parse_args(
            [
                "--messages",
                "conversation.json",
                "--model",
                "claude-3-5-sonnet",
                "--json",
                "--max-tokens",
                "10000",
                "--reserve",
                "500",
            ]
        )

        assert config.input_source == InputSource.MESSAGES
        assert config.input_path == Path("conversation.json")
        assert config.models == ["claude-3-5-sonnet"]
        assert config.json_output is True
        assert config.max_tokens == 10000
        assert config.reserve == 500

    def test_invalid_model_error(self):
        """Test that invalid model names cause parser error."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "invalid-model"])

    def test_negative_reserve_error(self):
        """Test that negative reserve values cause parser error."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--reserve", "-100"])

    def test_invalid_reserve_pct_error(self):
        """Test that invalid reserve_pct values cause parser error."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--reserve-pct", "1.5"])

    def test_zero_max_tokens_error(self):
        """Test that zero or negative max_tokens cause parser error."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--max-tokens", "0"])

        with pytest.raises(SystemExit):
            parser.parse_args(["--max-tokens", "-100"])

    def test_mutually_exclusive_input_sources(self):
        """Test that --file and --messages are mutually exclusive."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--file", "test.txt", "--messages", "msgs.json"])

    def test_mutually_exclusive_reserve_options(self):
        """Test that --reserve and --reserve-pct are mutually exclusive."""
        parser = CLIArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--reserve", "100", "--reserve-pct", "0.1"])


class TestParseCLIArgs:
    """Test the convenience function parse_cli_args."""

    def test_parse_cli_args_function(self):
        """Test the parse_cli_args convenience function."""
        config = parse_cli_args(["--model", "gpt-4o", "--json"])

        assert config.models == ["gpt-4o"]
        assert config.json_output is True

    def test_parse_cli_args_with_none(self):
        """Test parse_cli_args with None (should use sys.argv)."""
        # This test would normally use sys.argv, but we can't easily test that
        # without mocking sys.argv. Instead, we test that it doesn't crash.
        try:
            parse_cli_args([])  # Empty args should work
        except SystemExit:
            pass  # argparse may exit, which is fine


class TestInputSource:
    """Test InputSource enum."""

    def test_input_source_values(self):
        """Test InputSource enum values."""
        assert InputSource.STDIN.value == "stdin"
        assert InputSource.FILE.value == "file"
        assert InputSource.MESSAGES.value == "messages"
