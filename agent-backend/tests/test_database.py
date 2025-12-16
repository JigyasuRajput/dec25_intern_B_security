"""Tests for database and model functionality."""
import uuid
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from models import EmailEvent, EmailStatus, Organisation


@pytest.mark.asyncio
async def test_init_db_creates_tables():
    """Tables are created after init_db."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Verify tables exist
    async with AsyncSession(engine) as session:
        result = await session.exec(select(EmailEvent))
        emails = result.all()
        assert emails == []

    await engine.dispose()


@pytest.mark.asyncio
async def test_email_event_default_status(test_session: AsyncSession, test_org: dict):
    """New emails default to PENDING status."""
    email = EmailEvent(
        id=uuid.uuid4(),
        org_id=test_org["id"],
        sender="sender@test.com",
        recipient="recipient@test.com",
        subject="Default Status Test",
    )
    test_session.add(email)
    await test_session.commit()
    await test_session.refresh(email)

    assert email.status == EmailStatus.pending


@pytest.mark.asyncio
async def test_email_event_timestamps(test_session: AsyncSession, test_org: dict):
    """created_at/updated_at fields are set correctly."""
    before_create = datetime.utcnow()

    email = EmailEvent(
        id=uuid.uuid4(),
        org_id=test_org["id"],
        sender="sender@test.com",
        recipient="recipient@test.com",
        subject="Timestamp Test",
    )
    test_session.add(email)
    await test_session.commit()
    await test_session.refresh(email)

    after_create = datetime.utcnow()

    assert email.created_at is not None
    assert before_create <= email.created_at <= after_create
    assert email.updated_at is not None


@pytest.mark.asyncio
async def test_email_event_optional_fields(test_session: AsyncSession, test_org: dict):
    """Optional fields default to None."""
    email = EmailEvent(
        id=uuid.uuid4(),
        org_id=test_org["id"],
        sender="sender@test.com",
        recipient="recipient@test.com",
        subject="Optional Fields Test",
    )
    test_session.add(email)
    await test_session.commit()
    await test_session.refresh(email)

    assert email.body_preview is None
    assert email.risk_score is None
    assert email.risk_tier is None
    assert email.analysis_result is None


@pytest.mark.asyncio
async def test_organisation_crud(test_session: AsyncSession):
    """Create and read Organisation."""
    org = Organisation(
        id=uuid.uuid4(),
        name="Test Org",
        domain="testorg.com",
        api_key="test-key-abc",
    )
    test_session.add(org)
    await test_session.commit()

    result = await test_session.exec(
        select(Organisation).where(Organisation.api_key == "test-key-abc")
    )
    fetched = result.first()
    assert fetched is not None
    assert fetched.name == "Test Org"
