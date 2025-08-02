"""Utility functions shared across services."""

from typing import Optional


def truncate(text: Optional[str], limit: int = 200) -> str:
    """Return text truncated to a maximum of ``limit`` characters.

    Args:
        text: The string to truncate. ``None`` is treated as an empty string.
        limit: Maximum length of the returned string.

    Returns:
        The truncated string. If the original text exceeds ``limit``, the
        result will end with an ellipsis (``...``).
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    # Reserve space for ellipsis
    return text[: max(0, limit - 3)] + "..."
