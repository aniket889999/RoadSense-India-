"""Add safe usable capacity hazard associations, audit events, and pavement inspection records.

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. parking_hazard_associations
    op.create_table(
        "parking_hazard_associations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("parking_space_id", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False, server_default="BAY"),
        sa.Column("road_event_id", sa.String(length=64), nullable=True),
        sa.Column("hazard_label", sa.String(length=128), nullable=False),
        sa.Column("review_state", sa.String(length=32), nullable=False, server_default="UNREVIEWED"),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False, server_default="ACTIVE"),
        sa.Column("severity_label", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=False, server_default="operator"),
        sa.Column("reviewed_by", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parking_space_id"], ["parking_spaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["road_event_id"], ["road_events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_parking_hazard_associations_camera_id", "parking_hazard_associations", ["camera_id"])
    op.create_index("ix_parking_hazard_associations_parking_space_id", "parking_hazard_associations", ["parking_space_id"])
    op.create_index("ix_parking_hazard_associations_road_event_id", "parking_hazard_associations", ["road_event_id"])
    op.create_index("ix_parking_hazard_associations_review_state", "parking_hazard_associations", ["review_state"])
    op.create_index("ix_parking_hazard_associations_lifecycle_state", "parking_hazard_associations", ["lifecycle_state"])

    # 2. parking_hazard_audit_events
    op.create_table(
        "parking_hazard_audit_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("association_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("operator_identity", sa.String(length=128), nullable=False),
        sa.Column("prior_review_state", sa.String(length=32), nullable=True),
        sa.Column("new_review_state", sa.String(length=32), nullable=True),
        sa.Column("prior_lifecycle_state", sa.String(length=32), nullable=True),
        sa.Column("new_lifecycle_state", sa.String(length=32), nullable=True),
        sa.Column("explicit_reason", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["association_id"], ["parking_hazard_associations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_parking_hazard_audit_events_association_id", "parking_hazard_audit_events", ["association_id"])

    # 3. pavement_inspection_records
    op.create_table(
        "pavement_inspection_records",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("inspected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("inspector_label", sa.String(length=128), nullable=False, server_default="inspector"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["drive_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pavement_inspection_records_camera_id", "pavement_inspection_records", ["camera_id"])
    op.create_index("ix_pavement_inspection_records_session_id", "pavement_inspection_records", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_pavement_inspection_records_session_id", table_name="pavement_inspection_records")
    op.drop_index("ix_pavement_inspection_records_camera_id", table_name="pavement_inspection_records")
    op.drop_table("pavement_inspection_records")

    op.drop_index("ix_parking_hazard_audit_events_association_id", table_name="parking_hazard_audit_events")
    op.drop_table("parking_hazard_audit_events")

    op.drop_index("ix_parking_hazard_associations_lifecycle_state", table_name="parking_hazard_associations")
    op.drop_index("ix_parking_hazard_associations_review_state", table_name="parking_hazard_associations")
    op.drop_index("ix_parking_hazard_associations_road_event_id", table_name="parking_hazard_associations")
    op.drop_index("ix_parking_hazard_associations_parking_space_id", table_name="parking_hazard_associations")
    op.drop_index("ix_parking_hazard_associations_camera_id", table_name="parking_hazard_associations")
    op.drop_table("parking_hazard_associations")
