"""add_parking_layout_entities

Revision ID: d4e5f6a7b8c9
Revises: 80f60113d725
Create Date: 2026-09-06 11:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = '80f60113d725'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Sites table
    op.create_table(
        'sites',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('timezone', sa.String(length=64), nullable=False, server_default='UTC'),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 2. Cameras table
    op.create_table(
        'cameras',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('site_id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('reference_image_path', sa.String(length=512), nullable=True),
        sa.Column('reference_image_sha256', sa.String(length=64), nullable=True),
        sa.Column('reference_width', sa.Integer(), nullable=True),
        sa.Column('reference_height', sa.Integer(), nullable=True),
        sa.Column('calibration_status', sa.String(length=32), nullable=False, server_default='NOT_CONFIGURED'),
        sa.Column('camera_position_description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['site_id'], ['sites.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cameras_site_id'), 'cameras', ['site_id'], unique=False)

    # 3. ParkingLayoutRevision table
    op.create_table(
        'parking_layout_revisions',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('camera_id', sa.String(length=64), nullable=False),
        sa.Column('revision_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='DRAFT'),
        sa.Column('canonical_sha256', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('invalidated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('invalidation_reason', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_parking_layout_revisions_camera_id'), 'parking_layout_revisions', ['camera_id'], unique=False)

    # 4. ParkingSpace table
    op.create_table(
        'parking_spaces',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('layout_revision_id', sa.String(length=64), nullable=False),
        sa.Column('operator_label', sa.String(length=64), nullable=False),
        sa.Column('space_type', sa.String(length=32), nullable=False, server_default='STANDARD'),
        sa.Column('polygon_normalized', sa.JSON(), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['layout_revision_id'], ['parking_layout_revisions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_parking_spaces_layout_revision_id'), 'parking_spaces', ['layout_revision_id'], unique=False)

    # 5. ApproachZone table
    op.create_table(
        'approach_zones',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('parking_space_id', sa.String(length=64), nullable=False),
        sa.Column('polygon_normalized', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['parking_space_id'], ['parking_spaces.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('parking_space_id')
    )
    op.create_index(op.f('ix_approach_zones_parking_space_id'), 'approach_zones', ['parking_space_id'], unique=True)

    # 6. LayoutAuditEvent table
    op.create_table(
        'layout_audit_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('layout_revision_id', sa.String(length=64), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('prior_status', sa.String(length=32), nullable=True),
        sa.Column('new_status', sa.String(length=32), nullable=False),
        sa.Column('local_operator_label', sa.String(length=128), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['layout_revision_id'], ['parking_layout_revisions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_layout_audit_events_layout_revision_id'), 'layout_audit_events', ['layout_revision_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_layout_audit_events_layout_revision_id'), table_name='layout_audit_events')
    op.drop_table('layout_audit_events')
    op.drop_index(op.f('ix_approach_zones_parking_space_id'), table_name='approach_zones')
    op.drop_table('approach_zones')
    op.drop_index(op.f('ix_parking_spaces_layout_revision_id'), table_name='parking_spaces')
    op.drop_table('parking_spaces')
    op.drop_index(op.f('ix_parking_layout_revisions_camera_id'), table_name='parking_layout_revisions')
    op.drop_table('parking_layout_revisions')
    op.drop_index(op.f('ix_cameras_site_id'), table_name='cameras')
    op.drop_table('cameras')
    op.drop_table('sites')
