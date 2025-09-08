"""Main entry point for the token counter CLI."""

import sys
from typing import List


def main(args: List[str] = None) -> int:
    """Main entry point for the token counter CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0=success, 1=runtime error, 2=budget exceeded)
    """
    if args is None:
        args = sys.argv[1:]

    # TODO: Implement CLI logic in subsequent tasks
    print("Token Counter CLI - Not yet implemented")
    return 0


def cli_entry_point() -> None:
    """Console script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    cli_entry_point()
