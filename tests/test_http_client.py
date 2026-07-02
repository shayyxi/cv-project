from unittest.mock import Mock

from requests import Session

from app.ingestion.http_client import HTTPClient


def test_download_image() -> None:
    session = Mock(spec=Session)

    response = Mock()
    response.content = b"image-bytes"
    response.headers = {
        "Content-Type": "image/jpeg",
        "ETag": "abc123",
        "Last-Modified": "Mon, 01 Jan 2026 10:00:00 GMT",
    }
    response.raise_for_status.return_value = None

    session.get.return_value = response

    client = HTTPClient(session)

    image = client.download_image("https://example.com/test.jpg")

    assert image.url == "https://example.com/test.jpg"
    assert image.content == b"image-bytes"
    assert image.content_type == "image/jpeg"
    assert image.content_length == len(b"image-bytes")
    assert image.etag == "abc123"
    assert image.last_modified == "Mon, 01 Jan 2026 10:00:00 GMT"

    session.get.assert_called_once_with(
        "https://example.com/test.jpg",
        timeout=30,
    )