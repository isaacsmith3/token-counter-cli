"""Token counting functionality for different models."""

import time
from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from .input import InputData, Message
from .models import ModelDefinition


@dataclass
class CountingResult:
    """Result of token counting operation."""

    model: str
    input_tokens: int
    error: Optional[str] = None
    is_approximate: bool = False


class TokenCounter:
    """Handles token counting for different model types."""

    def count_tokens(
        self, input_data: InputData, model: ModelDefinition
    ) -> CountingResult:
        """Count tokens using appropriate strategy based on model type.

        Args:
            input_data: Input data to count tokens for
            model: Model definition specifying counting strategy

        Returns:
            CountingResult with token count or error information
        """
        try:
            if model.tokenizer_type == "local":
                return self._count_local_tokens(input_data, model)
            elif model.tokenizer_type == "provider":
                return self._count_provider_tokens(input_data, model)
            else:
                return CountingResult(
                    model=model.name,
                    input_tokens=0,
                    error=f"Unknown tokenizer type: {model.tokenizer_type}",
                )
        except Exception as e:
            return CountingResult(
                model=model.name,
                input_tokens=0,
                error=f"Token counting failed: {str(e)}",
            )

    def _count_local_tokens(
        self, input_data: InputData, model: ModelDefinition
    ) -> CountingResult:
        """Count tokens using local tiktoken for gpt-4o.

        Args:
            input_data: Input data to count
            model: Model definition (should be gpt-4o)

        Returns:
            CountingResult with local token count
        """
        if model.name == "gpt-4o":
            return self._count_gpt4o_tokens(input_data)
        else:
            return CountingResult(
                model=model.name,
                input_tokens=0,
                error=f"Local counting not supported for model: {model.name}",
            )

    def _count_gpt4o_tokens(self, input_data: InputData) -> CountingResult:
        """Count tokens for gpt-4o using tiktoken with o200k_base encoding.

        Args:
            input_data: Input data to count

        Returns:
            CountingResult with token count and approximation flag for messages
        """
        try:
            # Get the tiktoken encoding for gpt-4o (uses o200k_base)
            encoding = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            return CountingResult(
                model="gpt-4o",
                input_tokens=0,
                error=f"Failed to load tiktoken encoding: {str(e)}",
            )

        try:
            if input_data.messages is not None:
                # For structured messages, use approximation strategy
                token_count = self._count_messages_approximate(
                    input_data.messages, encoding
                )
                return CountingResult(
                    model="gpt-4o",
                    input_tokens=token_count,
                    is_approximate=True,
                )
            else:
                # For plain text, count directly
                token_count = len(encoding.encode(input_data.content))
                return CountingResult(
                    model="gpt-4o",
                    input_tokens=token_count,
                    is_approximate=False,
                )
        except Exception as e:
            return CountingResult(
                model="gpt-4o",
                input_tokens=0,
                error=f"Token encoding failed: {str(e)}",
            )

    def _count_messages_approximate(self, messages: List[Message], encoding) -> int:
        """Count tokens for messages using approximation strategy.

        This is an approximate method that concatenates message content with
        simple heuristics. It's documented as approximate because it doesn't
        account for the exact formatting tokens that OpenAI uses internally.

        Args:
            messages: List of messages to count
            encoding: tiktoken encoding to use

        Returns:
            Approximate token count
        """
        # Simple approximation: concatenate all text content with role prefixes
        # This is based on common patterns but is explicitly approximate
        text_parts = []

        for message in messages:
            # Add role as a prefix (approximate overhead)
            role_text = f"<{message.role}>"
            text_parts.append(role_text)

            # Extract and add content
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                # For array content, extract text where possible
                content_text = self._extract_text_from_content_array(message.content)
                if content_text:
                    text_parts.append(content_text)

        # Join with double newlines (approximate message separation)
        full_text = "\n\n".join(text_parts)

        # Count tokens for the approximated text
        return len(encoding.encode(full_text))

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

    def _count_provider_tokens(
        self, input_data: InputData, model: ModelDefinition
    ) -> CountingResult:
        """Count tokens using provider API (placeholder for future implementation).

        Args:
            input_data: Input data to count
            model: Model definition

        Returns:
            CountingResult indicating provider counting is not yet implemented
        """
        return CountingResult(
            model=model.name,
            input_tokens=0,
            error="Provider token counting not yet implemented",
        )


def count_tokens(input_data: InputData, model: ModelDefinition) -> CountingResult:
    """Count tokens using appropriate strategy based on model type.

    This is a convenience function that creates a TokenCounter instance
    and calls the count_tokens method.

    Args:
        input_data: Input data to count tokens for
        model: Model definition specifying counting strategy

    Returns:
        CountingResult with token count or error information
    """
    counter = TokenCounter()
    return counter.count_tokens(input_data, model)
