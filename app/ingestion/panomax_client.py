from app.dto import DownloadedImageDTO
from app.exceptions import ValidationError
from app.ingestion.http_client import HTTPClient


class PanomaxClient:
    BASE_URL = "https://live-image.panomax.com"

    def __init__(self, http_client: HTTPClient) -> None:
        self._http_client = http_client

    def build_latest_image_url(self, camera_id: str) -> str:
        return f"{self.BASE_URL}/cams/{camera_id}/recent_full.jpg"

    def download_latest_image(self, camera_id: str) -> DownloadedImageDTO:
        url = self.build_latest_image_url(camera_id)
        downloaded_image = self._http_client.download_image(url)

        if not downloaded_image.content:
            raise ValidationError(f"Empty image downloaded for camera_id={camera_id}")

        if downloaded_image.content_type and "image" not in downloaded_image.content_type:
            raise ValidationError(
                f"Invalid content type for camera_id={camera_id}: "
                f"{downloaded_image.content_type}"
            )

        return downloaded_image