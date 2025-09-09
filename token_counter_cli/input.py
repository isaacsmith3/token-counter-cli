"""Input handling for the token counter CLI."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

from .cli import CLIConfig, InputSource


@dataclass
class Message:
    """Represents a chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List]  # Allow both string and array content

    def __post_init__(self) -> None:
        """Validate message after initialization."""
        valid_roles = {"system", "user", "assistant", "tool"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid message role: {self.role}. "
                f"Valid roles: {', '.join(sorted(valid_roles))}"
            )


@dataclass
class InputData:
    """Container for input data with metadata."""

    content: str
    messages: Optional[List[Message]]
    source: str  # for error reporting


class InputHandler:
    """Handles reading input from various sources."""

    def read_input(self, config: CLIConfig) -> InputData:
        """Read input based on configuration.

        Args:
            config: CLI configuration specifying input source

        Returns:
            InputData containing content and optional messages

        Raises:
            FileNotFoundError: If specified file doesn't exist
            PermissionError: If file cannot be read
            ValueError: If JSON is invalid or messages have invalid format
            UnicodeDecodeError: If file cannot be decoded as UTF-8
        """
        if config.input_source == InputSource.MESSAGES:
            return self._read_messages_file(config.input_path)
        elif config.input_source == InputSource.FILE:
            return self._read_text_file(config.input_path)
        else:  # InputSource.STDIN
            return self._read_stdin()

    def _read_stdin(self) -> InputData:
        """Read text content from stdin.

        Returns:
            InputData with content from stdin

        Raises:
            UnicodeDecodeError: If stdin cannot be decoded as UTF-8
        """
        try:
            content = sys.stdin.read()
            return InputData(content=content, messages=None, source="stdin")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode stdin as UTF-8: {e.reason}",
            )

    def _read_text_file(self, file_path: Path) -> InputData:
        """Read text content from a file.

        Args:
            file_path: Path to the text file

        Returns:
            InputData with file content

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file cannot be read
            UnicodeDecodeError: If file cannot be decoded as UTF-8
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            return InputData(content=content, messages=None, source=str(file_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_path}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode file {file_path} as UTF-8: {e.reason}",
            )

    def _read_messages_file(self, file_path: Path) -> InputData:
        """Read and parse JSON messages file.

        Args:
            file_path: Path to the JSON messages file

        Returns:
            InputData with parsed messages and concatenated content

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file cannot be read
            UnicodeDecodeError: If file cannot be decoded as UTF-8
            ValueError: If JSON is invalid or messages have invalid format
        """
        try:
            # Read the file
            json_content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Messages file not found: {file_path}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading messages file: {file_path}"
            )
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Failed to decode messages file {file_path} as UTF-8: {e.reason}",
            )

        # Parse JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in messages file {file_path}: {e}")

        # Validate and parse messages
        messages = self.parse_messages(data, str(file_path))

        # Generate concatenated content for models that need plain text
        content = self._messages_to_text(messages)

        return InputData(content=content, messages=messages, source=str(file_path))

    def parse_messages(
        self, data: Union[List, dict], source: str = "input"
    ) -> List[Message]:
        """Parse and validate messages from JSON data.

        Args:
            data: JSON data (should be a list of message objects)
            source: Source description for error reporting

        Returns:
            List of validated Message objects

        Raises:
            ValueError: If data format is invalid or messages have invalid roles
        """
        if not isinstance(data, list):
            raise ValueError(
                f"Messages must be an array of objects in {source}, got {type(data).__name__}"
            )

        if not data:
            raise ValueError(f"Messages array cannot be empty in {source}")

        messages = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Message {i} must be an object in {source}, got {type(item).__name__}"
                )

            # Check required fields
            if "role" not in item:
                raise ValueError(
                    f"Message {i} missing required field 'role' in {source}"
                )
            if "content" not in item:
                raise ValueError(
                    f"Message {i} missing required field 'content' in {source}"
                )

            try:
                message = Message(role=item["role"], content=item["content"])
                messages.append(message)
            except ValueError as e:
                raise ValueError(f"Message {i} in {source}: {e}")

        return messages

    def _messages_to_text(self, messages: List[Message]) -> str:
        """Convert messages to concatenated text for plain text counting.

        Args:
            messages: List of messages to convert

        Returns:
            Concatenated text content with double newlines between messages
        """
        text_parts = []
        for message in messages:
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                # For array content, extract text where possible
                # This is a simple heuristic for MVP
                content_text = self._extract_text_from_content_array(message.content)
                if content_text:
                    text_parts.append(content_text)
            # Skip messages with no extractable text content

        return "\n\n".join(text_parts)

    def _extract_text_from_content_array(self, content_array: List) -> str:
        """Extract text from content array (simple heuristic for MVP).

        Args:
            content_array: Array of content items

        Returns:
            Extracted text content
        """
        text_parts = []
        for item in content_array:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                # Handle common pattern like {"type": "text", "text": "..."}
                if isinstance(item["text"], str):
                    text_parts.append(item["text"])

        return " ".join(text_parts)
