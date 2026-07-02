from app.storage.database import SessionLocal
from app.storage.repositories.image_job_repository import ImageJobRepository


def main() -> None:
    session = SessionLocal()

    try:
        repository = ImageJobRepository(session)

        deleted = repository.delete_all()

        print(f"Deleted {deleted} image jobs.")

    finally:
        session.close()


if __name__ == "__main__":
    main()