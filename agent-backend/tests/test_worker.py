"""Tests for worker functions."""
import uuid
from unittest.mock import patch

import pytest
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from models import EmailEvent, EmailStatus, RiskTier
from worker import (
    build_dummy_analysis,
    classify_risk,
    fetch_pending,
    process_email,
)


class TestClassifyRisk:
    """Tests for classify_risk function."""

    def test_classify_risk_safe_low(self):
        """Scores 0-29 return SAFE."""
        assert classify_risk(0) == RiskTier.safe
        assert classify_risk(15) == RiskTier.safe
        assert classify_risk(29) == RiskTier.safe

    def test_classify_risk_cautious(self):
        """Scores 30-79 return CAUTIOUS."""
        assert classify_risk(30) == RiskTier.cautious
        assert classify_risk(50) == RiskTier.cautious
        assert classify_risk(79) == RiskTier.cautious

    def test_classify_risk_threat(self):
        """Scores 80-100 return THREAT."""
        assert classify_risk(80) == RiskTier.threat
        assert classify_risk(90) == RiskTier.threat
        assert classify_risk(100) == RiskTier.threat


class TestBuildDummyAnalysis:
    """Tests for build_dummy_analysis function."""

    def test_build_dummy_analysis_structure(self):
        """Returns correct dict structure."""
        result = build_dummy_analysis(50)
        assert "indicators" in result
        assert "confidence" in result
        assert "threat_type" in result
        assert "analyzed_at" in result
        assert isinstance(result["indicators"], list)

    def test_build_dummy_analysis_confidence_bounds(self):
        """Confidence is bounded between 0 and 1."""
        result_low = build_dummy_analysis(0)
        assert result_low["confidence"] == 0.0

        result_high = build_dummy_analysis(100)
        assert result_high["confidence"] == 1.0

        result_mid = build_dummy_analysis(50)
        assert result_mid["confidence"] == 0.5

    def test_build_dummy_analysis_threat_type(self):
        """Threat type based on score threshold."""
        result_low = build_dummy_analysis(30)
        assert result_low["threat_type"] == "info"

        result_high = build_dummy_analysis(50)
        assert result_high["threat_type"] == "phishing"


class TestFetchPending:
    """Tests for fetch_pending function."""

    @pytest.mark.asyncio
    async def test_fetch_pending_returns_pending_only(
        self, test_session: AsyncSession, test_org: dict
    ):
        """Only fetches PENDING emails."""
        # Create emails with different statuses
        pending_email = EmailEvent(
            id=uuid.uuid4(),
            org_id=test_org["id"],
            sender="sender@test.com",
            recipient="recipient@test.com",
            subject="Pending Email",
            status=EmailStatus.pending,
        )
        completed_email = EmailEvent(
            id=uuid.uuid4(),
            org_id=test_org["id"],
            sender="sender@test.com",
            recipient="recipient@test.com",
            subject="Completed Email",
            status=EmailStatus.completed,
        )
        test_session.add(pending_email)
        test_session.add(completed_email)
        await test_session.commit()

        # Fetch pending
        result = await fetch_pending(test_session)

        # Only pending email should be returned
        assert len(result) >= 1
        assert all(e.status == EmailStatus.pending for e in result)
        assert any(e.subject == "Pending Email" for e in result)
        assert not any(e.subject == "Completed Email" for e in result)

    @pytest.mark.asyncio
    async def test_fetch_pending_respects_batch_limit(
        self, test_session: AsyncSession, test_org: dict
    ):
        """Respects BATCH_LIMIT setting."""
        # Create 15 pending emails
        for i in range(15):
            email = EmailEvent(
                id=uuid.uuid4(),
                org_id=test_org["id"],
                sender="sender@test.com",
                recipient="recipient@test.com",
                subject=f"Email {i}",
                status=EmailStatus.pending,
            )
            test_session.add(email)
        await test_session.commit()

        # Default BATCH_LIMIT is 10
        result = await fetch_pending(test_session)
        assert len(result) <= 10


class TestProcessEmail:
    """Tests for process_email function."""

    @pytest.mark.asyncio
    async def test_process_email_updates_status(
        self, test_session: AsyncSession, test_org: dict
    ):
        """Transitions email through PROCESSING to COMPLETED."""
        email = EmailEvent(
            id=uuid.uuid4(),
            org_id=test_org["id"],
            sender="sender@test.com",
            recipient="recipient@test.com",
            subject="Process Test",
            status=EmailStatus.pending,
        )
        test_session.add(email)
        await test_session.commit()
        await test_session.refresh(email)

        # Process the email
        await process_email(test_session, email)

        # Refresh from database
        result = await test_session.exec(
            select(EmailEvent).where(EmailEvent.id == email.id)
        )
        processed = result.first()
        assert processed is not None
        assert processed.status == EmailStatus.completed

    @pytest.mark.asyncio
    async def test_process_email_sets_risk_fields(
        self, test_session: AsyncSession, test_org: dict
    ):
        """Sets risk_score, risk_tier, analysis_result."""
        email = EmailEvent(
            id=uuid.uuid4(),
            org_id=test_org["id"],
            sender="sender@test.com",
            recipient="recipient@test.com",
            subject="Risk Fields Test",
            status=EmailStatus.pending,
        )
        test_session.add(email)
        await test_session.commit()
        await test_session.refresh(email)

        # Process the email
        await process_email(test_session, email)

        # Refresh from database
        result = await test_session.exec(
            select(EmailEvent).where(EmailEvent.id == email.id)
        )
        processed = result.first()
        assert processed is not None
        assert processed.risk_score is not None
        assert 0 <= processed.risk_score <= 100
        assert processed.risk_tier is not None
        assert processed.risk_tier in [RiskTier.safe, RiskTier.cautious, RiskTier.threat]
        assert processed.analysis_result is not None
        assert "indicators" in processed.analysis_result
