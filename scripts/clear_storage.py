import shutil

from app.config import settings


def remove(path):
    if path.exists():
        shutil.rmtree(path)


def main():
    remove(settings.local_raw_dir)
    remove(settings.local_processed_dir)
    remove(settings.local_failed_dir)
    remove(settings.local_analytics_dir)

    print("Storage cleared.")


if __name__ == "__main__":
    main()