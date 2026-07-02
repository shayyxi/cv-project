import logging

from app.ingestion.panomax_client import PanomaxClient
from app.storage.repositories.image_job_repository import ImageJobRepository
from app.storage.object_storage import ObjectStorage
from app.utils.clock import utc_now
from app.utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        camera_ids: list[str],
        panomax_client: PanomaxClient,
        storage: ObjectStorage,
        image_job_repository: ImageJobRepository,
    ) -> None:
        self._camera_ids = camera_ids
        self._panomax_client = panomax_client
        self._storage = storage
        self._image_job_repository = image_job_repository

    def poll(self) -> None:
        logger.info(
            "Starting ingestion cycle for %d cameras.",
            len(self._camera_ids),
        )

        successful = 0
        skipped = 0
        failed = 0

        for camera_id in self._camera_ids:
            try:
                is_new_image = self._poll_camera(camera_id)

                if is_new_image:
                    successful += 1
                else:
                    skipped += 1

            except Exception:
                failed += 1
                logger.exception(
                    "Polling failed for camera_id=%s",
                    camera_id,
                )

        logger.info(
            "Ingestion cycle completed. New=%d Skipped=%d Failed=%d",
            successful,
            skipped,
            failed,
        )

    def _poll_camera(self, camera_id: str) -> bool:
        downloaded_image = self._panomax_client.download_latest_image(
            camera_id=camera_id,
        )

        image_hash = sha256_bytes(downloaded_image.content)

        if not self._image_job_repository.is_new_image(
            camera_id=camera_id,
            image_hash=image_hash,
        ):
            logger.debug(
                "Duplicate image skipped for camera_id=%s",
                camera_id,
            )
            return False

        saved_path = self._storage.save_raw_image(
            camera_id=camera_id,
            image_bytes=downloaded_image.content,
        )

        logger.debug(
            "Saved raw image camera_id=%s path=%s",
            camera_id,
            saved_path,
        )

        self._image_job_repository.create_downloaded(
            camera_id=camera_id,
            remote_url=downloaded_image.url,
            raw_image_path=str(saved_path),
            sha256=image_hash,
            downloaded_at=utc_now(),
        )

        logger.info(
            "Downloaded new image camera_id=%s sha256=%s",
            camera_id,
            image_hash[:8],
        )

        return True