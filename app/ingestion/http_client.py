from requests import RequestException, Response, Session

from app.dto import DownloadedImageDTO
from app.exceptions import DownloadError


class HTTPClient:
    def __init__(self, session: Session) -> None:
        self._session = session

    def download_image(self, url: str) -> DownloadedImageDTO:
        try:
            response: Response = self._session.get(url, timeout=30)
            response.raise_for_status()
        except RequestException as exc:
            raise DownloadError(f"Failed to download image from {url}") from exc

        return DownloadedImageDTO(
            url=url,
            content=response.content,
            content_type=response.headers.get("Content-Type"),
            content_length=len(response.content),
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )