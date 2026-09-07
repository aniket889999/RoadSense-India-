import os
import shutil
import tempfile
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from services.api.app.db.base import Base
from services.api.app.db.session import get_db
import services.api.app.services.stability_job_manager as job_mgr_module
import services.api.app.services.parking_occupancy_job_manager as occ_job_mgr_module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_session(monkeypatch):
    tmp_dir = tempfile.mkdtemp(prefix="roadsense_test_db_")
    db_path = Path(tmp_dir) / "test.db"
    test_db_url = f"sqlite+aiosqlite:///{db_path}"

    engine = create_async_engine(
        test_db_url,
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30.0},
    )

    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    monkeypatch.setattr(job_mgr_module, "async_session_factory", session_factory)
    monkeypatch.setattr(occ_job_mgr_module, "async_session_factory", session_factory)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        yield session

    await engine.dispose()
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
async def client(db_session):
    from services.api.app.main import app

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
