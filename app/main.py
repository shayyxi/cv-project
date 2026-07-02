import argparse

from app.application.bootstrap import Application
from app.application.runner import ApplicationRunner
from app.utils.logging import configure_logging

def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single ingestion cycle.",
    )

    args = parser.parse_args()

    application = Application()

    runner = ApplicationRunner(
        pipeline=application.pipeline,
    )

    if args.run_once:
        runner.run_once()
    else:
        runner.run()


if __name__ == "__main__":
    main()