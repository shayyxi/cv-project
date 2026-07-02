from datetime import timezone

from app.utils.clock import utc_now


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()

    assert now.tzinfo == timezone.utc