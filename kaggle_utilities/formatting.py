"""Human-readable number formatting."""

from __future__ import annotations


def human_number(n: float) -> str:
    """Format a number as 1-3 digits with a suffix (K, M, B).

    Examples:
        500       -> "500"
        1_500     -> "1.5K"
        23_000    -> "23K"
        144_000_000 -> "144M"
        1_200_000_000 -> "1.2B"
        1_000_000_000_000 -> "1000B"
    """
    if abs(n) < 1_000:
        return f"{n:g}"
    for threshold, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(n) >= threshold:
            value = n / threshold
            if value >= 100:
                return f"{value:.0f}{suffix}"
            if value >= 10:
                s = f"{value:.1f}"
            else:
                s = f"{value:.2f}"
            # Strip trailing zeros after decimal point
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s + suffix
    return f"{n:g}"
