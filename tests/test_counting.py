"""Tests for token counting functionality."""

import json
import pytest
import tiktoken
from pathlib import Path
from unittest.mock import patch, MagicMock

from token_counter_cli.counting import TokenCounter, count_tokens, CountingResult
from token_counter_cli.input import InputData, Message
from token_counter_cli.models import ModelDefinition


class TestTokenCounter:
    """Test cases for TokenCounter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.gpt4o_model = ModelDefinition(
            name="gpt-4o", context_limit=128000, tokenizer_type="local"
        )
        self.claude_model = ModelDefinition(
            name="claude-3-5-sonnet", context_limit=200000, tokenizer_type="provider"
        )

    def test_count_tokens_plain_text_gpt4o(self):
        """Test counting tokens for plain text with gpt-4o."""
        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens > 0
        assert result.error is None
        assert result.is_approximate is False

    def test_count_tokens_messages_gpt4o(self):
        """Test counting tokens for messages with gpt-4o (approximate)."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there! How can I help you?"),
        ]
        input_data = InputData(content="", messages=messages, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens > 0
        assert result.error is None
        assert result.is_approximate is True

    def test_count_tokens_empty_text(self):
        """Test counting tokens for empty text."""
        input_data = InputData(content="", messages=None, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == 0
        assert result.error is None
        assert result.is_approximate is False

    def test_count_tokens_provider_not_implemented(self):
        """Test that provider counting returns not implemented error."""
        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, self.claude_model)

        assert result.model == "claude-3-5-sonnet"
        assert result.input_tokens == 0
        assert "not yet implemented" in result.error.lower()

    def test_count_tokens_unknown_tokenizer_type(self):
        """Test handling of unknown tokenizer type."""
        unknown_model = ModelDefinition(
            name="unknown-model", context_limit=1000, tokenizer_type="unknown"
        )
        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, unknown_model)

        assert result.model == "unknown-model"
        assert result.input_tokens == 0
        assert "Unknown tokenizer type" in result.error

    def test_count_tokens_unsupported_local_model(self):
        """Test handling of unsupported local model."""
        unsupported_model = ModelDefinition(
            name="unsupported-local", context_limit=1000, tokenizer_type="local"
        )
        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, unsupported_model)

        assert result.model == "unsupported-local"
        assert result.input_tokens == 0
        assert "Local counting not supported" in result.error

    @patch("tiktoken.encoding_for_model")
    def test_tiktoken_loading_error(self, mock_encoding):
        """Test handling of tiktoken loading errors."""
        mock_encoding.side_effect = Exception("Failed to load encoding")

        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == 0
        assert "Failed to load tiktoken encoding" in result.error

    @patch("tiktoken.encoding_for_model")
    def test_tiktoken_encoding_error(self, mock_encoding):
        """Test handling of tiktoken encoding errors."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = Exception("Encoding failed")
        mock_encoding.return_value = mock_enc

        input_data = InputData(content="Hello, world!", messages=None, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == 0
        assert "Token encoding failed" in result.error


class TestGPT4OTokenCounting:
    """Test cases specifically for GPT-4O token counting with golden strings."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.gpt4o_model = ModelDefinition(
            name="gpt-4o", context_limit=128000, tokenizer_type="local"
        )

    def test_golden_string_simple_text(self):
        """Test token counting for simple text (golden string test)."""
        # This is a deterministic test with a known token count
        text = "Hello, world!"
        input_data = InputData(content=text, messages=None, source="test")

        # Get expected count using tiktoken directly
        encoding = tiktoken.encoding_for_model("gpt-4o")
        expected_tokens = len(encoding.encode(text))

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == expected_tokens
        assert result.error is None
        assert result.is_approximate is False

    def test_golden_string_multiline_text(self):
        """Test token counting for multiline text (golden string test)."""
        text = "Line 1\nLine 2\nLine 3"
        input_data = InputData(content=text, messages=None, source="test")

        # Get expected count using tiktoken directly
        encoding = tiktoken.encoding_for_model("gpt-4o")
        expected_tokens = len(encoding.encode(text))

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == expected_tokens
        assert result.error is None
        assert result.is_approximate is False

    def test_golden_string_unicode_text(self):
        """Test token counting for Unicode text (golden string test)."""
        text = "Hello 世界! 🌍 Café naïve résumé"
        input_data = InputData(content=text, messages=None, source="test")

        # Get expected count using tiktoken directly
        encoding = tiktoken.encoding_for_model("gpt-4o")
        expected_tokens = len(encoding.encode(text))

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == expected_tokens
        assert result.error is None
        assert result.is_approximate is False

    def test_golden_string_code_text(self):
        """Test token counting for code text (golden string test)."""
        text = """def hello_world():
    print("Hello, world!")
    return 42"""
        input_data = InputData(content=text, messages=None, source="test")

        # Get expected count using tiktoken directly
        encoding = tiktoken.encoding_for_model("gpt-4o")
        expected_tokens = len(encoding.encode(text))

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens == expected_tokens
        assert result.error is None
        assert result.is_approximate is False

    def test_messages_approximation_consistency(self):
        """Test that message approximation is consistent."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
        ]
        input_data = InputData(content="", messages=messages, source="test")

        # Run the same test multiple times to ensure consistency
        results = []
        for _ in range(3):
            result = self.counter.count_tokens(input_data, self.gpt4o_model)
            results.append(result.input_tokens)

        # All results should be the same (deterministic)
        assert all(tokens == results[0] for tokens in results)
        assert all(result > 0 for result in results)

    def test_messages_with_array_content(self):
        """Test message counting with array content."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "url": "data:image/jpeg;base64,/9j/4AAQ..."},
                ],
            )
        ]
        input_data = InputData(content="", messages=messages, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens > 0  # Should extract text from array
        assert result.error is None
        assert result.is_approximate is True

    def test_messages_empty_content_array(self):
        """Test message counting with empty or non-text content array."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "image", "url": "data:image/jpeg;base64,/9j/4AAQ..."}
                ],
            )
        ]
        input_data = InputData(content="", messages=messages, source="test")

        result = self.counter.count_tokens(input_data, self.gpt4o_model)

        assert result.model == "gpt-4o"
        assert result.input_tokens >= 0  # May be 0 if no text extracted
        assert result.error is None
        assert result.is_approximate is True

    def test_golden_strings_from_file(self):
        """Test token counting against golden strings from fixture file."""
        # Load golden strings
        fixtures_path = Path(__file__).parent / "fixtures" / "golden_strings.json"
        if not fixtures_path.exists():
            pytest.skip("Golden strings fixture file not found")

        with open(fixtures_path) as f:
            golden_data = json.load(f)

        gpt4o_tests = golden_data.get("gpt-4o", {})

        for test_name, test_data in gpt4o_tests.items():
            input_text = test_data["input"]
            expected_tokens = test_data["expected_tokens"]

            input_data = InputData(content=input_text, messages=None, source="test")
            result = self.counter.count_tokens(input_data, self.gpt4o_model)

            assert result.model == "gpt-4o", f"Failed for test: {test_name}"
            assert (
                result.input_tokens == expected_tokens
            ), f"Failed for test: {test_name} - expected {expected_tokens}, got {result.input_tokens}"
            assert result.error is None, f"Failed for test: {test_name}"
            assert result.is_approximate is False, f"Failed for test: {test_name}"


class TestConvenienceFunction:
    """Test cases for the convenience function."""

    def test_count_tokens_function(self):
        """Test the convenience count_tokens function."""
        input_data = InputData(content="Hello, world!", messages=None, source="test")
        model = ModelDefinition(
            name="gpt-4o", context_limit=128000, tokenizer_type="local"
        )

        result = count_tokens(input_data, model)

        assert isinstance(result, CountingResult)
        assert result.model == "gpt-4o"
        assert result.input_tokens > 0
        assert result.error is None


class TestMessageApproximation:
    """Test cases for message approximation strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()

    def test_extract_text_from_content_array_string_items(self):
        """Test extracting text from array with string items."""
        content_array = ["Hello", "world", "!"]

        result = self.counter._extract_text_from_content_array(content_array)

        assert result == "Hello world !"

    def test_extract_text_from_content_array_dict_items(self):
        """Test extracting text from array with dict items."""
        content_array = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image", "url": "http://example.com/image.jpg"},
        ]

        result = self.counter._extract_text_from_content_array(content_array)

        assert result == "Hello world"

    def test_extract_text_from_content_array_mixed_items(self):
        """Test extracting text from array with mixed items."""
        content_array = [
            "Direct string",
            {"type": "text", "text": "Dict text"},
            {"type": "image", "url": "http://example.com/image.jpg"},
            {"other": "field"},
        ]

        result = self.counter._extract_text_from_content_array(content_array)

        assert result == "Direct string Dict text"

    def test_extract_text_from_content_array_empty(self):
        """Test extracting text from empty array."""
        content_array = []

        result = self.counter._extract_text_from_content_array(content_array)

        assert result == ""

    def test_extract_text_from_content_array_no_text(self):
        """Test extracting text from array with no text content."""
        content_array = [
            {"type": "image", "url": "http://example.com/image.jpg"},
            {"other": "field"},
        ]

        result = self.counter._extract_text_from_content_array(content_array)

        assert result == ""
