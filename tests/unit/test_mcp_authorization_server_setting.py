"""Unit tests — the ``MCP_AUTHORIZATION_SERVER`` setting (QA BUG-003).

The public protected-resource metadata used to derive ``authorization_servers``
inline from the raw ``LOGTO_URL``, advertising a container-internal address
(the QA stack's ``http://logto:3001``) or an empty string shaped into
``["/oidc"]``. The new explicit setting decouples the PUBLIC issuer from the
container-side JWKS base:

* an explicit value wins verbatim (the configmap sets the public Logto
  issuer);
* an unset/blank value falls back to ``<logto_url>/oidc``, staying empty when
  ``LOGTO_URL`` is empty too (nothing is advertised);
* a present-but-non-absolute value REFUSES startup (same
  present-but-invalid ⇒ refuse rule as the other configuration validators).
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


class TestMcpAuthorizationServerResolution:
    def test_explicit_value_is_kept_verbatim(self):
        settings = Settings(
            database_url=_DUMMY_URL,
            logto_url="http://logto:3001",
            mcp_authorization_server="https://logto.soludev.tech/oidc",
        )
        assert settings.mcp_authorization_server == "https://logto.soludev.tech/oidc"

    def test_unset_value_falls_back_to_the_logto_issuer(self):
        """QA BUG-003 fallback: ``<logto_url>/oidc`` (the historical document
        value) — dev/QA deployments keep advertising the same issuer."""
        settings = Settings(database_url=_DUMMY_URL, logto_url="http://logto:3001", mcp_authorization_server="")
        assert settings.mcp_authorization_server == "http://logto:3001/oidc"

    def test_unset_value_kept_empty_without_logto_url(self):
        """With neither value configured NOTHING is advertised (the route
        then serves ``authorization_servers: []``) — never a broken
        ``["/oidc"]`` shape."""
        settings = Settings(database_url=_DUMMY_URL, logto_url="", mcp_authorization_server="")
        assert settings.mcp_authorization_server == ""

    def test_blank_value_is_treated_as_unset(self):
        settings = Settings(database_url=_DUMMY_URL, logto_url="http://logto:3001", mcp_authorization_server="   ")
        assert settings.mcp_authorization_server == "http://logto:3001/oidc"

    def test_env_var_name_is_respected_and_wins_over_the_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """The env var is ``MCP_AUTHORIZATION_SERVER`` and an explicit value
        takes precedence over the logto-issuer fallback (the production
        configmap contract)."""
        monkeypatch.delenv("MCP_AUTHORIZATION_SERVER", raising=False)
        settings = Settings(database_url=_DUMMY_URL)
        assert settings.mcp_authorization_server

        monkeypatch.setenv("MCP_AUTHORIZATION_SERVER", "https://logto.soludev.tech/oidc")
        settings = Settings(database_url=_DUMMY_URL)
        assert settings.mcp_authorization_server == "https://logto.soludev.tech/oidc"


class TestMcpAuthorizationServerShape:
    def test_present_but_non_absolute_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="mcp_authorization_server"):
            Settings(database_url=_DUMMY_URL, mcp_authorization_server="logto:3001/oidc")

    def test_refusal_message_names_the_expected_shape(self):
        with pytest.raises(ValidationError, match="absolute URL"):
            Settings(database_url=_DUMMY_URL, mcp_authorization_server="example.invalid/oidc")


class TestMcpAuthorizationServerStartupWarnings:
    """Code-review NEW-3: the two startup warnings from the lifespan are part
    of the contract (QA BUG-003 "loud" half) — the mechanism that turns a
    silent misconfiguration into a visible one. Pinned through the REAL
    lifespan with caplog so a refactor that drops them fails a test."""

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

    async def test_empty_authorization_server_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(main_settings, "mcp_authorization_server", "")
        with caplog.at_level(logging.WARNING, logger="src.main"):
            await self._run_lifespan()
        assert any(
            LogMessage.MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED in record.getMessage() for record in caplog.records
        )

    async def test_non_https_authorization_server_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The warning interpolates the configured (non-https) issuer — assert
        the FULL formatted line."""
        monkeypatch.setattr(main_settings, "mcp_authorization_server", "http://logto:3001/oidc")
        expected = LogMessage.MCP_AUTHORIZATION_SERVER_NOT_HTTPS.replace("%s", "http://logto:3001/oidc")
        with caplog.at_level(logging.WARNING, logger="src.main"):
            await self._run_lifespan()
        assert any(expected in record.getMessage() for record in caplog.records)
