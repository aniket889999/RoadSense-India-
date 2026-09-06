"""Tests for database constraints, partial indexes, and Alembic migrations."""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from services.api.app.models.entities import Base, Camera, ParkingLayoutRevision, Site
from services.api.app.routers.parking import LayoutRevisionStatus


@pytest.fixture
def sqlite_db():
    tmp_dir = tempfile.mkdtemp(prefix="roadsense_db_test_")
    db_path = Path(tmp_dir) / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session, engine

    session.close()
    engine.dispose()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unique_constraint_camera_revision_number(sqlite_db):
    """STEP 4.1: (camera_id, revision_number) must be unique."""
    session, engine = sqlite_db

    site = Site(id=str(uuid.uuid4()), name="Test Site")
    session.add(site)
    session.commit()

    camera = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Camera")
    session.add(camera)
    session.commit()

    rev1 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=1,
        status=LayoutRevisionStatus.DRAFT.value,
        reference_image_sha256="a" * 64,
        reference_width=640,
        reference_height=480,
    )
    session.add(rev1)
    session.commit()

    # Attempt to insert another revision with the same revision_number for the same camera
    rev2 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=1,  # Duplicate revision number
        status=LayoutRevisionStatus.DRAFT.value,
        reference_image_sha256="b" * 64,
        reference_width=640,
        reference_height=480,
    )
    session.add(rev2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_partial_unique_index_one_verified_layout_per_camera(sqlite_db):
    """STEP 4.2: Enforce at most one VERIFIED layout per camera at database level."""
    session, engine = sqlite_db

    site = Site(id=str(uuid.uuid4()), name="Test Site 2")
    session.add(site)
    session.commit()

    camera = Camera(id=str(uuid.uuid4()), site_id=site.id, name="Test Camera 2")
    session.add(camera)
    session.commit()

    rev1 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=1,
        status=LayoutRevisionStatus.VERIFIED.value,
        reference_image_sha256="a" * 64,
        reference_width=640,
        reference_height=480,
    )
    session.add(rev1)
    session.commit()

    # Attempt to insert a second VERIFIED revision for the same camera
    rev2 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=2,
        status=LayoutRevisionStatus.VERIFIED.value,  # Second VERIFIED
        reference_image_sha256="a" * 64,
        reference_width=640,
        reference_height=480,
    )
    session.add(rev2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    # But multiple INVALIDATED or DRAFT revisions are permitted
    rev3 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=3,
        status=LayoutRevisionStatus.INVALIDATED.value,
        reference_image_sha256="a" * 64,
        reference_width=640,
        reference_height=480,
    )
    rev4 = ParkingLayoutRevision(
        id=str(uuid.uuid4()),
        camera_id=camera.id,
        revision_number=4,
        status=LayoutRevisionStatus.INVALIDATED.value,
        reference_image_sha256="a" * 64,
        reference_width=640,
        reference_height=480,
    )
    session.add_all([rev3, rev4])
    session.commit()
    assert session.query(ParkingLayoutRevision).count() == 3


def test_alembic_upgrade_downgrade_cycle():
    """Verify Alembic migration upgrade to head, downgrade -1, upgrade to head on a disposable database."""
    from alembic import command
    from alembic.config import Config

    tmp_dir = tempfile.mkdtemp(prefix="roadsense_alembic_test_")
    db_path = Path(tmp_dir) / "alembic_test.db"
    db_url = f"sqlite:///{db_path}"

    alembic_ini_path = Path(__file__).resolve().parent.parent / "services" / "api" / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        # Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Downgrade one revision
        command.downgrade(alembic_cfg, "-1")

        # Re-upgrade to head
        command.upgrade(alembic_cfg, "head")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
