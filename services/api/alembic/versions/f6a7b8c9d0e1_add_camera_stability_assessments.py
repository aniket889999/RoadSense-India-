"""add_camera_stability_assessments

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-09-06 15:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'camera_stability_assessments',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('camera_id', sa.String(length=64), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('layout_revision_id', sa.String(length=64), sa.ForeignKey('parking_layout_revisions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('layout_canonical_sha256', sa.String(length=64), nullable=True),
        sa.Column('reference_image_sha256', sa.String(length=64), nullable=False),
        sa.Column('video_sha256', sa.String(length=64), nullable=False),
        sa.Column('algorithm_version', sa.String(length=64), nullable=False),
        sa.Column('opencv_version', sa.String(length=32), nullable=False),
        sa.Column('thresholds_snapshot', sa.JSON(), nullable=False),
        sa.Column('sample_measurements', sa.JSON(), nullable=False),
        sa.Column('aggregate_decision', sa.String(length=32), nullable=False),
        sa.Column('operational_gate', sa.String(length=32), nullable=False),
        sa.Column('gate_reasons', sa.JSON(), nullable=False),
        sa.Column('summary_metrics', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('operator_acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('operator_label', sa.String(length=128), nullable=True),
        sa.Column('operator_note', sa.Text(), nullable=True),
    )
    op.create_index('ix_camera_stability_camera_id', 'camera_stability_assessments', ['camera_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_camera_stability_camera_id', table_name='camera_stability_assessments')
    op.drop_table('camera_stability_assessments')
