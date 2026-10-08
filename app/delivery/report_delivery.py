import logging
import smtplib
from collections.abc import Iterable
from email.message import EmailMessage
from pathlib import Path

import requests

from app.config import settings

logger = logging.getLogger(__name__)


class ReportDeliveryService:
    """
    Delivers generated report files to a webhook (one POST per file)
    or by email (one message with every file attached).
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

        if not url or not api_key:
            raise ValueError(
                "Webhook delivery is not configured; set "
                "WORDPRESS_WEBHOOK_URL and WORDPRESS_API_KEY in .env"
            )

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
        report_paths: str | Path | Iterable[str | Path],
        subject: str = "PPE compliance report",
    ) -> None:
        """
        Email the report(s) as attachments of one message via the
        configured SMTP server (SMTP_HOST/PORT/USER/PASSWORD/TO in
        .env). Accepts a single path or any iterable of paths, e.g.
        the site-wide report followed by the per-camera reports.
        """

        if not settings.smtp_host:
            raise ValueError(
                "SMTP is not configured; set SMTP_HOST, SMTP_USER, "
                "SMTP_PASSWORD and SMTP_TO in .env"
            )

        if isinstance(report_paths, (str, Path)):
            report_paths = [report_paths]

        paths = [Path(path) for path in report_paths]

        if not paths:
            raise ValueError("No report files to email")

        message = EmailMessage()
        message["From"] = settings.smtp_user
        message["To"] = settings.smtp_to
        message["Subject"] = subject
        message.set_content(
            "Attached: automated PPE compliance report"
            + (
                "s (site-wide first, then one per camera)."
                if len(paths) > 1
                else "."
            )
        )

        for path in paths:
            message.add_attachment(
                path.read_bytes(),
                maintype="application",
                subtype="pdf",
                filename=path.name,
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
            "%d report file(s) emailed -> %s",
            len(paths),
            settings.smtp_to,
        )
