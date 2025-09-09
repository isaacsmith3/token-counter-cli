"""Tests for input handling functionality."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from token_counter_cli.cli import CLIConfig, InputSource
from token_counter_cli.input import InputData, InputHandler, Message


class TestMessage:
    """Tests for Message dataclass."""

    def test_valid_message_creation(self):
        """Test creating valid messages."""
        message = Message(role="user", content="Hello world")
        assert message.role == "user"
        assert message.content == "Hello world"

    def test_valid_roles(self):
        """Test all valid message roles."""
        valid_roles = ["system", "user", "assistant", "tool"]
        for role in valid_roles:
            message = Message(role=role, content="test")
            assert message.role == role

    def test_invalid_role(self):
        """Test invalid message role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid message role: invalid"):
            Message(role="invalid", content="test")

    def test_array_content(self):
        """Test message with array content."""
        content = [{"type": "text", "text": "Hello"}, {"type": "image", "url": "..."}]
        message = Message(role="user", content=content)
        assert message.content == content


class TestInputHandler:
    """Tests for InputHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = InputHandler()

    def test_read_stdin(self):
        """Test reading from stdin."""
        test_input = "Hello from stdin"
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with patch.object(sys, "stdin", StringIO(test_input)):
            result = self.handler.read_input(config)

        assert result.content == test_input
        assert result.messages is None
        assert result.source == "stdin"

    def test_read_stdin_unicode_error(self):
        """Test stdin with unicode decode error."""
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.STDIN,
            input_path=None,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        # Mock stdin.read() to raise UnicodeDecodeError
        def mock_read():
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        with patch.object(sys.stdin, "read", mock_read):
            with pytest.raises(
                UnicodeDecodeError, match="Failed to decode stdin as UTF-8"
            ):
                self.handler.read_input(config)

    def test_read_text_file(self, tmp_path):
        """Test reading from text file."""
        test_content = "Hello from file"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content, encoding="utf-8")

        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.FILE,
            input_path=test_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        result = self.handler.read_input(config)

        assert result.content == test_content
        assert result.messages is None
        assert result.source == str(test_file)

    def test_read_text_file_not_found(self):
        """Test reading non-existent file."""
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.FILE,
            input_path=Path("nonexistent.txt"),
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with pytest.raises(FileNotFoundError, match="File not found: nonexistent.txt"):
            self.handler.read_input(config)

    def test_read_text_file_unicode_error(self, tmp_path):
        """Test reading file with unicode decode error."""
        test_file = tmp_path / "bad_encoding.txt"
        test_file.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.FILE,
            input_path=test_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with pytest.raises(
            UnicodeDecodeError, match=f"Failed to decode file {test_file} as UTF-8"
        ):
            self.handler.read_input(config)

    def test_read_messages_file_valid(self, tmp_path):
        """Test reading valid messages file."""
        messages_data = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages_file = tmp_path / "messages.json"
        messages_file.write_text(json.dumps(messages_data), encoding="utf-8")

        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.MESSAGES,
            input_path=messages_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        result = self.handler.read_input(config)

        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helpful assistant."
        assert result.content == "You are a helpful assistant.\n\nHello!\n\nHi there!"
        assert result.source == str(messages_file)

    def test_read_messages_file_not_found(self):
        """Test reading non-existent messages file."""
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.MESSAGES,
            input_path=Path("nonexistent.json"),
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with pytest.raises(
            FileNotFoundError, match="Messages file not found: nonexistent.json"
        ):
            self.handler.read_input(config)

    def test_read_messages_file_invalid_json(self, tmp_path):
        """Test reading messages file with invalid JSON."""
        messages_file = tmp_path / "invalid.json"
        messages_file.write_text("{ invalid json", encoding="utf-8")

        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.MESSAGES,
            input_path=messages_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with pytest.raises(
            ValueError, match=f"Invalid JSON in messages file {messages_file}"
        ):
            self.handler.read_input(config)

    def test_parse_messages_valid(self):
        """Test parsing valid messages."""
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        messages = self.handler.parse_messages(data)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there"

    def test_parse_messages_not_list(self):
        """Test parsing messages with non-list data."""
        data = {"role": "user", "content": "Hello"}

        with pytest.raises(ValueError, match="Messages must be an array of objects"):
            self.handler.parse_messages(data)

    def test_parse_messages_empty_list(self):
        """Test parsing empty messages list."""
        data = []

        with pytest.raises(ValueError, match="Messages array cannot be empty"):
            self.handler.parse_messages(data)

    def test_parse_messages_invalid_item_type(self):
        """Test parsing messages with non-object items."""
        data = ["not an object"]

        with pytest.raises(ValueError, match="Message 0 must be an object"):
            self.handler.parse_messages(data)

    def test_parse_messages_missing_role(self):
        """Test parsing messages missing role field."""
        data = [{"content": "Hello"}]

        with pytest.raises(ValueError, match="Message 0 missing required field 'role'"):
            self.handler.parse_messages(data)

    def test_parse_messages_missing_content(self):
        """Test parsing messages missing content field."""
        data = [{"role": "user"}]

        with pytest.raises(
            ValueError, match="Message 0 missing required field 'content'"
        ):
            self.handler.parse_messages(data)

    def test_parse_messages_invalid_role(self):
        """Test parsing messages with invalid role."""
        data = [{"role": "invalid", "content": "Hello"}]

        with pytest.raises(
            ValueError, match="Message 0 in input: Invalid message role: invalid"
        ):
            self.handler.parse_messages(data)

    def test_parse_messages_array_content(self):
        """Test parsing messages with array content."""
        data = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "url": "image.jpg"},
                ],
            }
        ]

        messages = self.handler.parse_messages(data)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert isinstance(messages[0].content, list)
        assert len(messages[0].content) == 2

    def test_messages_to_text_string_content(self):
        """Test converting messages with string content to text."""
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant response"),
        ]

        result = self.handler._messages_to_text(messages)

        assert result == "System prompt\n\nUser message\n\nAssistant response"

    def test_messages_to_text_array_content(self):
        """Test converting messages with array content to text."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            ),
            Message(role="assistant", content="Response"),
        ]

        result = self.handler._messages_to_text(messages)

        assert result == "Hello World\n\nResponse"

    def test_messages_to_text_mixed_content(self):
        """Test converting messages with mixed content types."""
        messages = [
            Message(role="user", content="Simple text"),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Array text"},
                    {"type": "image", "url": "image.jpg"},  # Should be ignored
                ],
            ),
            Message(role="assistant", content="Response"),
        ]

        result = self.handler._messages_to_text(messages)

        assert result == "Simple text\n\nArray text\n\nResponse"

    def test_extract_text_from_content_array_text_objects(self):
        """Test extracting text from content array with text objects."""
        content_array = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
            {"type": "image", "url": "image.jpg"},  # Should be ignored
        ]

        result = self.handler._extract_text_from_content_array(content_array)

        assert result == "Hello World"

    def test_extract_text_from_content_array_strings(self):
        """Test extracting text from content array with direct strings."""
        content_array = ["Hello", "World", 123]  # Non-string should be ignored

        result = self.handler._extract_text_from_content_array(content_array)

        assert result == "Hello World"

    def test_extract_text_from_content_array_empty(self):
        """Test extracting text from empty content array."""
        content_array = []

        result = self.handler._extract_text_from_content_array(content_array)

        assert result == ""

    def test_extract_text_from_content_array_no_text(self):
        """Test extracting text from content array with no text items."""
        content_array = [
            {"type": "image", "url": "image.jpg"},
            {"type": "audio", "url": "audio.mp3"},
        ]

        result = self.handler._extract_text_from_content_array(content_array)

        assert result == ""


class TestInputHandlerIntegration:
    """Integration tests for InputHandler with different input sources."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = InputHandler()

    def test_stdin_vs_file_precedence(self, tmp_path):
        """Test that --file takes precedence over stdin (requirement 1.3)."""
        # This test verifies the precedence logic is handled by CLI config
        # The InputHandler should respect the config's input_source

        file_content = "File content"
        test_file = tmp_path / "test.txt"
        test_file.write_text(file_content, encoding="utf-8")

        # Config should specify FILE source when --file is provided
        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.FILE,  # CLI should set this when --file is used
            input_path=test_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        # Even if stdin has content, file should be read
        stdin_content = "Stdin content that should be ignored"
        with patch.object(sys, "stdin", StringIO(stdin_content)):
            result = self.handler.read_input(config)

        assert result.content == file_content
        assert result.source == str(test_file)

    def test_complex_messages_file(self, tmp_path):
        """Test reading complex messages file with various content types."""
        messages_data = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image", "url": "data:image/jpeg;base64,/9j/4AAQ..."},
                ],
            },
            {
                "role": "assistant",
                "content": "I can see a beautiful landscape with mountains and a lake.",
            },
            {"role": "user", "content": "Can you describe it in more detail?"},
        ]

        messages_file = tmp_path / "complex_messages.json"
        messages_file.write_text(json.dumps(messages_data), encoding="utf-8")

        config = CLIConfig(
            models=["claude-3-5-sonnet"],
            input_source=InputSource.MESSAGES,
            input_path=messages_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        result = self.handler.read_input(config)

        # Verify messages are parsed correctly
        assert len(result.messages) == 4
        assert result.messages[0].role == "system"
        assert isinstance(result.messages[1].content, list)
        assert result.messages[2].role == "assistant"
        assert result.messages[3].role == "user"

        # Verify text concatenation extracts text content
        expected_text = (
            "You are a helpful assistant that can analyze images.\n\n"
            "What do you see in this image?\n\n"
            "I can see a beautiful landscape with mountains and a lake.\n\n"
            "Can you describe it in more detail?"
        )
        assert result.content == expected_text

    def test_error_propagation_with_source_info(self, tmp_path):
        """Test that errors include source information for debugging."""
        # Test with messages file to verify source is included in error messages
        messages_file = tmp_path / "error_messages.json"
        messages_data = [{"role": "invalid_role", "content": "This should fail"}]
        messages_file.write_text(json.dumps(messages_data), encoding="utf-8")

        config = CLIConfig(
            models=["gpt-4o"],
            input_source=InputSource.MESSAGES,
            input_path=messages_file,
            max_tokens=None,
            reserve=None,
            reserve_pct=0.2,
            json_output=False,
        )

        with pytest.raises(ValueError) as exc_info:
            self.handler.read_input(config)

        # Error should include the source file path
        error_message = str(exc_info.value)
        assert str(messages_file) in error_message
        assert "invalid_role" in error_message
