"""Add parking occupancy jobs table for gated video processing.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-07 15:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "parking_occupancy_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("site_id", sa.String(length=64), nullable=False),
        sa.Column("layout_revision_id", sa.String(length=64), nullable=True),
        sa.Column("stability_assessment_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="QUEUED"),
        sa.Column("progress_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("stage_message", sa.String(length=256), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("gate_decision", sa.String(length=32), nullable=True),
        sa.Column("gate_reasons", sa.JSON(), nullable=True),
        sa.Column("input_video_sha256", sa.String(length=64), nullable=True),
        sa.Column("output_video_sha256", sa.String(length=64), nullable=True),
        sa.Column("reference_image_sha256", sa.String(length=64), nullable=True),
        sa.Column("layout_canonical_sha256", sa.String(length=64), nullable=True),
        sa.Column("detector_checkpoint_sha256", sa.String(length=64), nullable=True),
        sa.Column("occupancy_config_sha256", sa.String(length=64), nullable=True),
        sa.Column("total_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fps", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("duration_seconds", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("video_width", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("video_height", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bays", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_occupied_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_vacant_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_unknown_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_occluded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_state_transitions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_video_path", sa.String(length=512), nullable=True),
        sa.Column("timeline_jsonl_path", sa.String(length=512), nullable=True),
        sa.Column("manifest_json", sa.JSON(), nullable=True),
        sa.Column("bay_summary_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["layout_revision_id"], ["parking_layout_revisions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["stability_assessment_id"], ["camera_stability_assessments.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_parking_occupancy_jobs_camera_id", "parking_occupancy_jobs", ["camera_id"])
    op.create_index("ix_parking_occupancy_jobs_site_id", "parking_occupancy_jobs", ["site_id"])


def downgrade() -> None:
    op.drop_index("ix_parking_occupancy_jobs_site_id", table_name="parking_occupancy_jobs")
    op.drop_index("ix_parking_occupancy_jobs_camera_id", table_name="parking_occupancy_jobs")
    op.drop_table("parking_occupancy_jobs")
