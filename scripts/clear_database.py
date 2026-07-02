from sqlalchemy import text

from app.storage.database import SessionLocal


def main() -> None:
    session = SessionLocal()

    try:
        session.execute(
            text("TRUNCATE TABLE image_jobs RESTART IDENTITY CASCADE")
        )
        session.commit()

        print("Database cleared successfully.")

    finally:
        session.close()


if __name__ == "__main__":
    main()