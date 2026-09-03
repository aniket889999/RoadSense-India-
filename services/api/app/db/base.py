"""SQLAlchemy Declarative Base for RoadSense India models."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass
