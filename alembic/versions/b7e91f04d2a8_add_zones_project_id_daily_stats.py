"""add zones table and daily_stats view

Revision ID: b7e91f04d2a8
Revises: 26c023d2c0e2
Create Date: 2026-08-31 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e91f04d2a8'
down_revision: Union[str, Sequence[str], None] = '26c023d2c0e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# One row per day per camera, counting person detections only
# (PPE item rows carry no compliance flags and must not be counted).
# project_id is an alias of camera_id: in this deployment each
# camera is its own project.
DAILY_STATS_VIEW = """
CREATE VIEW daily_stats AS
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


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'zones',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('camera_id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('polygon', sa.Text(), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_zones_camera_id'), 'zones', ['camera_id'], unique=False)

    op.execute(DAILY_STATS_VIEW)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP VIEW IF EXISTS daily_stats")

    op.drop_index(op.f('ix_zones_camera_id'), table_name='zones')
    op.drop_table('zones')
