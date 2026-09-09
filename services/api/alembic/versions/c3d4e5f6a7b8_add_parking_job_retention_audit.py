"""Add parking job retention audit table.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-08 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "parking_job_audit_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("operator_identity_assertion", sa.String(length=128), nullable=False),
        sa.Column("deletion_reason", sa.Text(), nullable=False),
        sa.Column("prior_status", sa.String(length=32), nullable=False),
        sa.Column("resulting_status", sa.String(length=32), nullable=False),
        sa.Column("artifact_hashes", sa.JSON(), nullable=True),
        sa.Column("purge_result", sa.String(length=64), nullable=False),
        sa.Column("request_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_time", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["parking_occupancy_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_parking_job_audit_events_job_id", "parking_job_audit_events", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_parking_job_audit_events_job_id", table_name="parking_job_audit_events")
    op.drop_table("parking_job_audit_events")
