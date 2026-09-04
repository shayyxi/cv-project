"""materialize daily and weekly stats views

Revision ID: c4d82a1f9e37
Revises: b7e91f04d2a8
Create Date: 2026-08-31 17:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4d82a1f9e37'
down_revision: Union[str, Sequence[str], None] = 'b7e91f04d2a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DAILY_STATS_SELECT = """
SELECT
    CAST(COALESCE(j.captured_at, j.downloaded_at, j.created_at) AS DATE) AS day,
    j.camera_id AS project_id,
    j.camera_id,
    COUNT(*) AS workers,
    COUNT(*) FILTER (WHERE d.is_compliant IS TRUE) AS compliant,
    COUNT(*) FILTER (WHERE d.helmet_compliant IS FALSE) AS helmet_viol,
    COUNT(*) FILTER (WHERE d.vest_compliant IS FALSE) AS vest_viol,
    COUNT(*) FILTER (WHERE d.boots_compliant IS FALSE) AS boots_viol
FROM detections d
JOIN image_jobs j ON j.id = d.image_job_id
WHERE d.label = 'person'
GROUP BY
    CAST(COALESCE(j.captured_at, j.downloaded_at, j.created_at) AS DATE),
    j.camera_id
"""

WEEKLY_STATS_SELECT = """
SELECT
    CAST(date_trunc('week', day) AS DATE) AS period,
    camera_id AS project_id,
    camera_id,
    SUM(workers) AS workers,
    SUM(compliant) AS compliant,
    SUM(helmet_viol) AS helmet_viol,
    SUM(vest_viol) AS vest_viol,
    SUM(boots_viol) AS boots_viol
FROM daily_stats
GROUP BY
    CAST(date_trunc('week', day) AS DATE),
    camera_id
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("DROP VIEW IF EXISTS daily_stats")

    op.execute(
        f"CREATE MATERIALIZED VIEW daily_stats AS {DAILY_STATS_SELECT}"
    )
    op.execute(
        "CREATE INDEX ix_daily_stats_day ON daily_stats (day)"
    )

    op.execute(
        f"CREATE MATERIALIZED VIEW weekly_stats AS {WEEKLY_STATS_SELECT}"
    )
    op.execute(
        "CREATE INDEX ix_weekly_stats_period ON weekly_stats (period)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS weekly_stats")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS daily_stats")

    op.execute(f"CREATE VIEW daily_stats AS {DAILY_STATS_SELECT}")
