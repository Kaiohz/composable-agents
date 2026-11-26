"""Unit tests — the ``MCP_PUBLIC_BASE_URL`` setting (code-review NEW-1 / NEW-3).

The advertised discovery origin is the pointer the whole OAuth2 discovery
chain starts from:

* a present-but-SCHEME-LESS value used to be accepted and published a
  malformed ``resource_metadata`` URL (``Bearer
  resource_metadata="host/.well-known/..."``) no MCP client can fetch — the
  chain broke SILENTLY. It now REFUSES startup (the same present-but-invalid
  ⇒ refuse rule as the sibling ``MCP_AUTHORIZATION_SERVER`` validator);
* present values are NORMALIZED at the Settings level (surrounding whitespace
  and a trailing slash stripped — the single normalization source);
* a present ``http://`` value stays non-fatal (dev/QA deployments) but is
  WARNED at startup through the centralized ``MCP_PUBLIC_BASE_URL_NOT_HTTPS``
  log message (pinned through the REAL lifespan with caplog).
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.domain.logging.messages import LogMessage
from src.main import app as main_app
from src.main import lifespan as main_lifespan
from src.main import settings as main_settings

_DUMMY_URL = "postgresql://user:pass@host:5432/db"


class TestMcpPublicBaseUrlNormalization:
    def test_trailing_slash_is_stripped_at_settings_level(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_public_base_url="https://composables-agent.soludev.tech/")
        assert settings.mcp_public_base_url == "https://composables-agent.soludev.tech"

    def test_surrounding_whitespace_is_stripped(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_public_base_url="  https://host.example.com/  ")
        assert settings.mcp_public_base_url == "https://host.example.com"

    def test_http_value_is_accepted_for_dev(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_public_base_url="http://composables.local:8000")
        assert settings.mcp_public_base_url == "http://composables.local:8000"

    def test_blank_value_is_treated_as_unset(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_public_base_url="   ")
        assert settings.mcp_public_base_url == ""


class TestMcpPublicBaseUrlRefusal:
    """Present-but-invalid ⇒ refuse startup (the sibling validator's rule)."""

    def test_scheme_less_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="mcp_public_base_url"):
            Settings(database_url=_DUMMY_URL, mcp_public_base_url="composables-agent.soludev.tech")

    def test_scheme_relative_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="mcp_public_base_url"):
            Settings(database_url=_DUMMY_URL, mcp_public_base_url="//composables-agent.soludev.tech")

    def test_refusal_message_names_the_absolute_url_shape(self):
        with pytest.raises(ValidationError, match="absolute URL"):
            Settings(database_url=_DUMMY_URL, mcp_public_base_url="host.example.com/mcp")

    def test_refusal_message_explains_the_silently_broken_discovery_chain(self):
        with pytest.raises(ValidationError, match="malformed resource_metadata"):
            Settings(database_url=_DUMMY_URL, mcp_public_base_url="example.com")


class TestMcpPublicBaseUrlNotHttpsWarning:
    """The warn half (non-fatal): an ``http://`` origin is warned through the
    REAL lifespan via the centralized ``MCP_PUBLIC_BASE_URL_NOT_HTTPS`` message
    (no test asserted the startup warnings before — code-review NEW-3). A
    configured https:// origin (the production shape) and an UNSET origin (the
    Host-allow-list fallback) must NOT warn."""

    async def _run_lifespan(self) -> None:
        """Drive the real lifespan with the external boundaries mocked off
        (the ``test_tracing_lifecycle`` pattern)."""
        with (
            patch("src.main.close_persistence", AsyncMock()),
            patch("src.main.init_persistence", AsyncMock()),
            patch("src.main.mcp_tool_loader", AsyncMock()),
            patch("src.main.tracing_provider", AsyncMock()),
        ):
            async with main_lifespan(main_app):
                pass

    async def test_non_https_value_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(main_settings, "mcp_public_base_url", "http://composables.local:8000")
        with caplog.at_level(logging.WARNING, logger="src.main"):
            await self._run_lifespan()
        assert any(LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage() for record in caplog.records)

    async def test_https_value_does_not_warn(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        monkeypatch.setattr(main_settings, "mcp_public_base_url", "https://composables-agent.soludev.tech")
        with caplog.at_level(logging.WARNING, logger="src.main"):
            await self._run_lifespan()
        assert not any(LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage() for record in caplog.records)

    async def test_unset_value_does_not_warn(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
        monkeypatch.setattr(main_settings, "mcp_public_base_url", "")
        with caplog.at_level(logging.WARNING, logger="src.main"):
            await self._run_lifespan()
        assert not any(LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage() for record in caplog.records)
