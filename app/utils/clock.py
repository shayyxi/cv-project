from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Returns the current UTC time as a timezone-aware datetime.
    """
    return datetime.now(timezone.utc)