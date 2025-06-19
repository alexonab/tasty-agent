import os


def is_test_env() -> bool:
    """Return True if TASTYTRADE_IS_TEST is set to a truthy value."""
    return os.getenv("TASTYTRADE_IS_TEST", "false").lower() in ("true", "1", "yes")

