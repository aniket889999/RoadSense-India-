"""enforce_parking_calibration_lifecycle

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-06 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch_alter_table for SQLite compatibility
    with op.batch_alter_table('parking_layout_revisions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('reference_image_sha256', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('reference_width', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('reference_height', sa.Integer(), nullable=True))
        batch_op.create_unique_constraint('uq_parking_layout_camera_revision', ['camera_id', 'revision_number'])

    # Partial unique index enforcing at most one VERIFIED layout per camera
    op.create_index(
        'uq_one_verified_layout_per_camera',
        'parking_layout_revisions',
        ['camera_id'],
        unique=True,
        postgresql_where=sa.text("status = 'VERIFIED'"),
        sqlite_where=sa.text("status = 'VERIFIED'"),
    )


def downgrade() -> None:
    op.drop_index('uq_one_verified_layout_per_camera', table_name='parking_layout_revisions')

    with op.batch_alter_table('parking_layout_revisions', schema=None) as batch_op:
        batch_op.drop_constraint('uq_parking_layout_camera_revision', type_='unique')
        batch_op.drop_column('reference_height')
        batch_op.drop_column('reference_width')
        batch_op.drop_column('reference_image_sha256')
