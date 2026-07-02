from app.application.bootstrap import Application
from app.utils.logging import configure_logging

def main():
    configure_logging()
    app = Application()

    app.pipeline.run_cycle()


if __name__ == "__main__":
    main()