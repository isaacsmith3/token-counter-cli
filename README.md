# Token Counter CLI

Cross-model token counting command-line tool that provides accurate token counts for multiple LLM models in a single run.

## Installation

```bash
pip install token-counter-cli
```

## Usage

```bash
# Count tokens from stdin for default models (gpt-4o, claude-3-5-sonnet)
echo "Hello world" | tc

# Count tokens from a file
tc --file input.txt

# Count tokens from structured messages
tc --messages messages.json

# Output in JSON format
tc --json < input.txt
```

## Supported Models

- **gpt-4o**: Local counting via tiktoken (approximate for messages)
- **claude-3-5-sonnet**: Provider counting via Anthropic API (in the future)

## Environment Variables

- `ANTHROPIC_API_KEY`: Required for claude-3-5-sonnet counting
- `NO_COLOR`: Disables ANSI color output

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy token_counter_cli
```

## License

MIT
