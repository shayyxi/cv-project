import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from app.analytics.analytics_config import load_analytics_config
from app.analytics.analytics_repository import AnalyticsRepository
from app.config import settings

logger = logging.getLogger(__name__)


class HeatmapService:
    """
    Density of non-compliant workers for one camera over a time
    window, rendered as a heat overlay on a reference frame from
    that camera (or as a raw JSON grid for a dashboard).
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
    ) -> None:
        self._repository = repository

        config = load_analytics_config()["heatmap"]

        self._window_days = int(config["window_days"])
        self._opacity = float(config["opacity"])

        self._output_dir = settings.local_analytics_dir

    def violation_heatmap(
        self,
        camera_id: str,
        days: int | None = None,
        reference_image: str | Path | None = None,
        as_json: bool = False,
        grid: tuple[int, int] = (60, 40),
    ) -> Path | None:
        """
        Returns the path of the written heatmap file, or None when
        no violations are logged for the camera and window.
        """

        days = days or self._window_days

        since = datetime.utcnow() - timedelta(days=days)

        boxes = self._repository.violation_boxes(
            camera_id=camera_id,
            since=since,
        )

        if not boxes:
            logger.info(
                "No violations logged for camera_id=%s in last %d days.",
                camera_id,
                days,
            )
            return None

        reference_path = (
            Path(reference_image)
            if reference_image is not None
            else self._latest_raw_image(camera_id)
        )

        image = cv2.imread(str(reference_path))

        if image is None:
            raise ValueError(
                f"Could not read reference image: {reference_path}"
            )

        height, width = image.shape[:2]

        self._output_dir.mkdir(parents=True, exist_ok=True)

        if as_json:
            return self._write_grid(
                camera_id,
                boxes,
                width,
                height,
                grid,
                days,
            )

        return self._write_overlay(
            camera_id,
            boxes,
            image,
        )

    # ==================================================================
    # Internals
    # ==================================================================

    def _latest_raw_image(self, camera_id: str) -> Path:
        camera_dir = settings.local_raw_dir / str(camera_id)

        images = sorted(camera_dir.glob("*.jpg"))

        if not images:
            raise ValueError(
                f"No raw images on disk for camera_id={camera_id}; "
                "pass reference_image explicitly."
            )

        return images[-1]

    def _write_overlay(
        self,
        camera_id: str,
        boxes: list[dict],
        image: np.ndarray,
    ) -> Path:
        height, width = image.shape[:2]

        # Heat accumulates where workers stood (bottom-center of box).
        heat = np.zeros((height, width), dtype=np.float32)

        for box in boxes:
            foot_x = int((box["x_min"] + box["x_max"]) / 2)
            foot_y = int(box["y_max"])

            if 0 <= foot_x < width and 0 <= foot_y < height:
                heat[foot_y, foot_x] += 1.0

        radius = max(40, width // 80)

        kernel = radius * 4 + 1

        heat = cv2.GaussianBlur(
            heat,
            (kernel, kernel),
            radius,
        )

        heat /= heat.max()

        colored = cv2.applyColorMap(
            (heat * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        mask = heat > 0.05

        overlay = image.copy()

        overlay[mask] = (
            image[mask] * (1.0 - self._opacity)
            + colored[mask] * self._opacity
        ).astype(np.uint8)

        output_path = self._output_dir / f"heatmap_{camera_id}.jpg"

        cv2.imwrite(
            str(output_path),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        logger.info(
            "Heatmap (%d violations) -> %s",
            len(boxes),
            output_path,
        )

        return output_path

    def _write_grid(
        self,
        camera_id: str,
        boxes: list[dict],
        width: int,
        height: int,
        grid: tuple[int, int],
        days: int,
    ) -> Path:
        grid_w, grid_h = grid

        heat = np.zeros((grid_h, grid_w), dtype=np.float32)

        for box in boxes:
            center_x = (box["x_min"] + box["x_max"]) / 2
            center_y = (box["y_min"] + box["y_max"]) / 2

            if 0 <= center_x < width and 0 <= center_y < height:
                heat[
                    int(center_y / height * grid_h),
                    int(center_x / width * grid_w),
                ] += 1

        output_path = self._output_dir / f"heatmap_{camera_id}.json"

        output_path.write_text(
            json.dumps(
                {
                    "camera_id": str(camera_id),
                    "days": days,
                    "grid_w": grid_w,
                    "grid_h": grid_h,
                    "grid": heat.tolist(),
                }
            ),
            encoding="utf-8",
        )

        logger.info(
            "Heatmap JSON grid (%d violations) -> %s",
            len(boxes),
            output_path,
        )

        return output_path
