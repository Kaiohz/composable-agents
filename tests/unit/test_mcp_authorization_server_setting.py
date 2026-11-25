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

import pytest
from pydantic import ValidationError

from src.config import Settings

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
