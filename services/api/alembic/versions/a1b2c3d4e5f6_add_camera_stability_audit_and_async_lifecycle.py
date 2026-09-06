"""add_camera_stability_audit_and_async_lifecycle

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-09-06 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add asynchronous lifecycle, progress, and provenance columns to camera_stability_assessments
    with op.batch_alter_table('camera_stability_assessments', schema=None) as batch_op:
        batch_op.add_column(sa.Column('status', sa.String(length=32), server_default='COMPLETE', nullable=False))
        batch_op.add_column(sa.Column('progress_pct', sa.Float(), server_default='100.0', nullable=False))
        batch_op.add_column(sa.Column('stage_message', sa.String(length=256), nullable=True))
        batch_op.add_column(sa.Column('failure_code', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('failure_message', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('config_version', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('config_sha256', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.alter_column('video_sha256', existing_type=sa.String(length=64), nullable=True)
        batch_op.alter_column('algorithm_version', existing_type=sa.String(length=64), nullable=True)
        batch_op.alter_column('opencv_version', existing_type=sa.String(length=32), nullable=True)
        batch_op.alter_column('thresholds_snapshot', existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column('sample_measurements', existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column('aggregate_decision', existing_type=sa.String(length=32), nullable=True)
        batch_op.alter_column('operational_gate', existing_type=sa.String(length=32), nullable=True)
        batch_op.alter_column('gate_reasons', existing_type=sa.JSON(), nullable=True)
        batch_op.alter_column('summary_metrics', existing_type=sa.JSON(), nullable=True)

    # 2. Create append-only camera_stability_audit_events table
    op.create_table(
        'camera_stability_audit_events',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('assessment_id', sa.String(length=64), sa.ForeignKey('camera_stability_assessments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('camera_id', sa.String(length=64), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('operator_identity', sa.String(length=128), nullable=False),
        sa.Column('explicit_reason', sa.Text(), nullable=False),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('previous_gate_state', sa.String(length=32), nullable=True),
        sa.Column('resulting_gate_state', sa.String(length=32), nullable=False),
        sa.Column('previous_calibration_status', sa.String(length=32), nullable=True),
        sa.Column('resulting_calibration_status', sa.String(length=32), nullable=False),
        sa.Column('assessment_sha256', sa.String(length=64), nullable=True),
        sa.Column('config_sha256', sa.String(length=64), nullable=True),
        sa.Column('reference_image_sha256', sa.String(length=64), nullable=False),
        sa.Column('layout_canonical_sha256', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_stability_audit_assessment_id', 'camera_stability_audit_events', ['assessment_id'], unique=False)
    op.create_index('ix_stability_audit_camera_id', 'camera_stability_audit_events', ['camera_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_stability_audit_camera_id', table_name='camera_stability_audit_events')
    op.drop_index('ix_stability_audit_assessment_id', table_name='camera_stability_audit_events')
    op.drop_table('camera_stability_audit_events')

    with op.batch_alter_table('camera_stability_assessments', schema=None) as batch_op:
        batch_op.drop_column('completed_at')
        batch_op.drop_column('started_at')
        batch_op.drop_column('config_sha256')
        batch_op.drop_column('config_version')
        batch_op.drop_column('failure_message')
        batch_op.drop_column('failure_code')
        batch_op.drop_column('stage_message')
        batch_op.drop_column('progress_pct')
        batch_op.drop_column('status')
