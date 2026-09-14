"""Add a real approach-zone reference to parking hazard associations.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-09-14 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, None] = "d5e6f7a8b9c0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("parking_hazard_associations") as batch_op:
        batch_op.add_column(sa.Column("approach_zone_id", sa.String(length=64), nullable=True))
        batch_op.create_foreign_key(
            "fk_parking_hazard_associations_approach_zone_id",
            "approach_zones",
            ["approach_zone_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "ix_parking_hazard_associations_approach_zone_id",
            ["approach_zone_id"],
        )

    # Migrate valid legacy approach-zone rows to the actual zone. Legacy rows
    # without a configured zone are conservatively retained as BAY hazards.
    op.execute(
        sa.text(
            "UPDATE parking_hazard_associations "
            "SET approach_zone_id = ("
            "SELECT approach_zones.id FROM approach_zones "
            "WHERE approach_zones.parking_space_id = "
            "parking_hazard_associations.parking_space_id"
            ") WHERE target_type = 'APPROACH_ZONE'"
        )
    )
    op.execute(
        sa.text(
            "UPDATE parking_hazard_associations SET target_type = 'BAY' "
            "WHERE target_type = 'APPROACH_ZONE' AND approach_zone_id IS NULL"
        )
    )

    with op.batch_alter_table("parking_hazard_associations") as batch_op:
        batch_op.create_check_constraint(
            "ck_parking_hazard_target_reference",
            "(target_type = 'BAY' AND approach_zone_id IS NULL) OR "
            "(target_type = 'APPROACH_ZONE' AND approach_zone_id IS NOT NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("parking_hazard_associations") as batch_op:
        batch_op.drop_constraint("ck_parking_hazard_target_reference", type_="check")
        batch_op.drop_index("ix_parking_hazard_associations_approach_zone_id")
        batch_op.drop_constraint(
            "fk_parking_hazard_associations_approach_zone_id", type_="foreignkey"
        )
        batch_op.drop_column("approach_zone_id")
