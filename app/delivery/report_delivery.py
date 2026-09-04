import logging
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class ReportDeliveryService:
    """
    Delivers a generated report file to a webhook or by email.
    """

    def deliver_webhook(
        self,
        report_path: str | Path,
        url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> int:
        """
        POST the report file to a webhook endpoint.

        Defaults to the configured WordPress webhook.
        Returns the HTTP status code.
        """

        url = url or settings.wordpress_webhook_url
        api_key = api_key or settings.wordpress_api_key

        report_path = Path(report_path)

        with report_path.open("rb") as f:
            response = requests.post(
                url,
                files={
                    "file": (report_path.name, f),
                },
                headers={
                    "X-API-Key": api_key,
                },
                timeout=timeout,
            )

        logger.info(
            "Report webhook %s -> HTTP %d",
            url,
            response.status_code,
        )

        return response.status_code

    def deliver_email(
        self,
        report_path: str | Path,
        subject: str = "PPE compliance report",
    ) -> None:
        """
        Email the report as an attachment via the configured
        SMTP server (SMTP_HOST/PORT/USER/PASSWORD/TO in .env).
        """

        if not settings.smtp_host:
            raise ValueError(
                "SMTP is not configured; set SMTP_HOST, SMTP_USER, "
                "SMTP_PASSWORD and SMTP_TO in .env"
            )

        report_path = Path(report_path)

        message = EmailMessage()
        message["From"] = settings.smtp_user
        message["To"] = settings.smtp_to
        message["Subject"] = subject
        message.set_content(
            "Attached: automated PPE compliance report."
        )

        message.add_attachment(
            report_path.read_bytes(),
            maintype="application",
            subtype="pdf",
            filename=report_path.name,
        )

        with smtplib.SMTP(
            settings.smtp_host,
            settings.smtp_port,
        ) as server:
            server.starttls()
            server.login(
                settings.smtp_user,
                settings.smtp_password,
            )
            server.send_message(message)

        logger.info(
            "Report emailed -> %s",
            settings.smtp_to,
        )
