from collections import Counter

from sqlalchemy import select

from app.storage.database import SessionLocal
from app.storage.models import ImageJob


def main() -> None:
    session = SessionLocal()

    try:
        jobs = (
            session.execute(
                select(ImageJob).order_by(ImageJob.downloaded_at.desc())
            )
            .scalars()
            .all()
        )

        print("\n========== IMAGE JOBS ==========\n")

        if not jobs:
            print("No image jobs found.")
            return

        for job in jobs:
            print(
                f"{job.camera_id:<8} "
                f"{job.status.value:<12} "
                f"{job.downloaded_at} "
                f"{job.sha256[:8]}"
            )

        print("\n========== SUMMARY ==========\n")

        print(f"Total Jobs : {len(jobs)}")

        camera_counter = Counter(job.camera_id for job in jobs)

        print("\nImages per Camera")

        for camera_id in sorted(camera_counter):
            print(
                f"  {camera_id:<8} {camera_counter[camera_id]}"
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()