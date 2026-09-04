"""drop zones table (image cropper polygon covers the zone use case)

Revision ID: d9e0f1a2b3c4
Revises: c4d82a1f9e37
Create Date: 2026-08-31 19:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd9e0f1a2b3c4'
down_revision: Union[str, Sequence[str], None] = 'c4d82a1f9e37'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index(op.f('ix_zones_camera_id'), table_name='zones')
    op.drop_table('zones')


def downgrade() -> None:
    """Downgrade schema."""
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
