from dataclasses import dataclass


@dataclass(frozen=True)
class DownloadedImageDTO:
    url: str
    content: bytes
    content_type: str | None
    content_length: int | None
    etag: str | None
    last_modified: str | None