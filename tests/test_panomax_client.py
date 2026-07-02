from unittest.mock import Mock

from app.dto import DownloadedImageDTO
from app.ingestion.panomax_client import PanomaxClient

import pytest

from app.exceptions import ValidationError


def test_build_latest_image_url() -> None:
    http_client = Mock()
    client = PanomaxClient(http_client=http_client)

    url = client.build_latest_image_url("6168")

    assert url == "https://live-image.panomax.com/cams/6168/recent_thumb.jpg"


def test_download_latest_image() -> None:
    http_client = Mock()
    http_client.download_image.return_value = DownloadedImageDTO(
        url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        content=b"image-bytes",
        content_type="image/jpeg",
        content_length=11,
        etag=None,
        last_modified=None,
    )

    client = PanomaxClient(http_client=http_client)

    result = client.download_latest_image("6168")

    assert result.content == b"image-bytes"
    http_client.download_image.assert_called_once_with(
        "https://live-image.panomax.com/cams/6168/recent_thumb.jpg"
    )

def test_download_latest_image_rejects_empty_content() -> None:
    http_client = Mock()
    http_client.download_image.return_value = DownloadedImageDTO(
        url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        content=b"",
        content_type="image/jpeg",
        content_length=0,
        etag=None,
        last_modified=None,
    )

    client = PanomaxClient(http_client=http_client)

    with pytest.raises(ValidationError):
        client.download_latest_image("6168")