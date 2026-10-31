from pathlib import Path

import pytest

from app.config import settings
from app.delivery import report_delivery
from app.delivery.report_delivery import ReportDeliveryService


class FakeSMTP:
    """Stands in for smtplib.SMTP; records every message sent."""

    sent: list = []

    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def starttls(self) -> None:
        pass

    def login(self, user, password) -> None:
        pass

    def send_message(self, message) -> None:
        FakeSMTP.sent.append(message)


@pytest.fixture()
def smtp(monkeypatch: pytest.MonkeyPatch):
    FakeSMTP.sent = []

    monkeypatch.setattr(report_delivery.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "bot@example.com")
    monkeypatch.setattr(settings, "smtp_password", "secret")
    monkeypatch.setattr(settings, "smtp_to", "safety@example.com")

    return FakeSMTP


def _pdf(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def _attachment_names(message) -> list[str]:
    return [part.get_filename() for part in message.iter_attachments()]


def test_email_attaches_every_report_to_one_message(smtp, tmp_path: Path) -> None:
    site = _pdf(tmp_path, "report_2026-09-07.pdf")
    camera = _pdf(tmp_path, "report_2026-09-07_camera_12990.pdf")

    ReportDeliveryService().deliver_email([site, camera])

    (message,) = smtp.sent

    assert message["To"] == "safety@example.com"
    assert message["From"] == "bot@example.com"
    assert _attachment_names(message) == [site.name, camera.name]


def test_email_accepts_a_single_path(smtp, tmp_path: Path) -> None:
    site = _pdf(tmp_path, "report_2026-09-07.pdf")

    ReportDeliveryService().deliver_email(site)

    (message,) = smtp.sent

    assert _attachment_names(message) == [site.name]


def test_email_refuses_an_empty_list(smtp) -> None:
    with pytest.raises(ValueError):
        ReportDeliveryService().deliver_email([])


def test_email_requires_smtp_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "smtp_host", "")

    with pytest.raises(ValueError):
        ReportDeliveryService().deliver_email(_pdf(tmp_path, "r.pdf"))


def test_webhook_unconfigured_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "wordpress_webhook_url", "")

    with pytest.raises(ValueError):
        ReportDeliveryService().deliver_webhook(_pdf(tmp_path, "r.pdf"))


def test_webhook_posts_the_file_with_the_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        settings, "wordpress_webhook_url", "https://example.com/hook"
    )
    monkeypatch.setattr(settings, "wordpress_api_key", "key")

    posted: dict = {}

    class Response:
        status_code = 201

    def fake_post(url, files, headers, timeout):
        posted.update(url=url, filename=files["file"][0], headers=headers)
        return Response()

    monkeypatch.setattr(report_delivery.requests, "post", fake_post)

    status = ReportDeliveryService().deliver_webhook(
        _pdf(tmp_path, "report.pdf")
    )

    assert status == 201
    assert posted == {
        "url": "https://example.com/hook",
        "filename": "report.pdf",
        "headers": {"X-API-Key": "key"},
    }
