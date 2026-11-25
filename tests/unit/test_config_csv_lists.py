"""Unit tests — Settings CSV list parsing (feat-mcp-oauth2 REQ-003/REQ-013).

Covers the ``jwt_audiences`` / ``mcp_allowed_hosts`` Settings seam added to
``src/config.py``, including the code-review findings:

* M-1 — a present-but-empty ``MCP_ALLOWED_HOSTS`` (``""`` or blank-only) must
  REFUSE startup (an empty host allow-list with DNS-rebinding protection
  ENABLED rejects every MCP request with HTTP 421 — a silent total outage);
* M-4 — the CSV parsing contract (trimming, list passthrough, defaults) and
  the legacy ``JWT_AUDIENCES``-unset tolerance (empty → ``[]`` — the JWT
  adapter then collapses to the legacy single-audience behavior).
"""

import pytest
from pydantic import ValidationError

from src.config import Settings

_DUMMY_URL = "postgresql://user:pass@host:5432/db"


class TestMcpAllowedHostsCsv:
    def test_present_but_empty_env_value_refuses_startup(self, monkeypatch: pytest.MonkeyPatch):
        """M-1: `MCP_ALLOWED_HOSTS=""` is INVALID — the process must refuse to
        start instead of silently 421ing every MCP request."""
        monkeypatch.setenv("MCP_ALLOWED_HOSTS", "")
        with pytest.raises(ValidationError, match="mcp_allowed_hosts"):
            Settings(database_url=_DUMMY_URL)

    def test_present_blank_only_env_value_refuses_startup(self, monkeypatch: pytest.MonkeyPatch):
        """M-1: `MCP_ALLOWED_HOSTS=" , "` (blank-only) is invalid too — the
        CSV split may not silently collapse into an empty allow-list."""
        monkeypatch.setenv("MCP_ALLOWED_HOSTS", " , ")
        with pytest.raises(ValidationError, match="mcp_allowed_hosts"):
            Settings(database_url=_DUMMY_URL)

    def test_refusal_message_names_the_outage_and_the_fix(self):
        """The refusal message must point the operator at the requirement."""
        with pytest.raises(ValidationError, match="DNS-rebinding"):
            Settings(database_url=_DUMMY_URL, mcp_allowed_hosts=",")

    def test_explicit_empty_value_refuses_startup(self):
        """The empty value is rejected regardless of how it reaches Settings."""
        with pytest.raises(ValidationError):
            Settings(database_url=_DUMMY_URL, mcp_allowed_hosts="")

    def test_csv_values_are_trimmed(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_allowed_hosts=" host-a , host-b ")
        assert settings.mcp_allowed_hosts == ["host-a", "host-b"]

    def test_list_values_pass_through_uncoerced(self):
        settings = Settings(database_url=_DUMMY_URL, mcp_allowed_hosts=["host-a:1"])
        assert settings.mcp_allowed_hosts == ["host-a:1"]

    def test_absent_value_falls_back_to_localhost_default(self, monkeypatch: pytest.MonkeyPatch):
        """An ABSENT env var keeps the dev/QA-safe localhost default."""
        monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
        settings = Settings(database_url=_DUMMY_URL)
        assert settings.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]


class TestJwtAudiencesCsv:
    def test_absent_value_falls_back_to_empty_list(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-003 backward-compat half: `JWT_AUDIENCES` unset → `[]` — the JWT
        adapter then keeps the exact legacy single-`JWT_AUDIENCE` behavior."""
        monkeypatch.delenv("JWT_AUDIENCES", raising=False)
        settings = Settings(database_url=_DUMMY_URL)
        assert settings.jwt_audiences == []

    def test_present_empty_value_tolerated_as_legacy_only(self):
        """An EMPTY `JWT_AUDIENCES` is legal (unlike MCP_ALLOWED_HOSTS): it
        simply means "legacy JWT_AUDIENCE only"."""
        settings = Settings(database_url=_DUMMY_URL, jwt_audiences="")
        assert settings.jwt_audiences == []

    def test_present_blank_only_value_tolerated_as_legacy_only(self):
        settings = Settings(database_url=_DUMMY_URL, jwt_audiences=" , ")
        assert settings.jwt_audiences == []

    def test_csv_values_are_trimmed(self):
        """Trimming applies per comma-separated part."""
        settings = Settings(database_url=_DUMMY_URL, jwt_audiences=" a , b ")
        assert settings.jwt_audiences == ["a", "b"]

    def test_duplicates_preserved_at_settings_level(self):
        """Duplication is handled by the JWT adapter's merge (legacy value
        first) — the Settings layer only splits and trims."""
        settings = Settings(database_url=_DUMMY_URL, jwt_audiences="a,b,a")
        assert settings.jwt_audiences == ["a", "b", "a"]

    def test_list_values_pass_through_uncoerced(self):
        settings = Settings(database_url=_DUMMY_URL, jwt_audiences=["a"])
        assert settings.jwt_audiences == ["a"]


class TestMcpAllowedHostsJsonStyleRefusal:
    """Review NEW-2: a JSON-style value must REFUSE startup with a message
    pointing the configmap author at the comma-separated form.

    The production configmap sets its neighbouring list key ``ALLOWED_ORIGINS``
    in JSON style (``'["https://composables.soludev.tech"]'``) — a copy-paste of
    that style into ``MCP_ALLOWED_HOSTS`` (or ``JWT_AUDIENCES``, which shares
    the same CSV seam) produced ONE garbage entry that matched nothing: a
    silent, total outage. Refuse instead (mirroring the M-1 empty-value
    refusal pattern).
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mcp_allowed_hosts", '["composables-agent.soludev.tech"]'),
            ("mcp_allowed_hosts", '["host-a", "host-b"]'),
            ("mcp_allowed_hosts", '{"host-a": true}'),
            ("mcp_allowed_hosts", "'host-a'"),
            ("jwt_audiences", '["ui-app-id"]'),
            ("jwt_audiences", '{"aud": true}'),
        ],
    )
    def test_json_style_value_refuses_startup(self, field: str, value: str):
        with pytest.raises(ValidationError, match="comma-separated"):
            Settings(database_url=_DUMMY_URL, **{field: value})

    def test_refusal_message_names_the_field_and_the_comma_form(self):
        with pytest.raises(
            ValidationError,
            match="mcp_allowed_hosts.*JSON-style.*MCP_ALLOWED_HOSTS=host-a,host-b",
        ):
            Settings(database_url=_DUMMY_URL, mcp_allowed_hosts='["host-a"]')

    def test_refusal_message_works_for_the_audiences_field_too(self):
        with pytest.raises(ValidationError, match="jwt_audiences received a JSON-style value"):
            Settings(database_url=_DUMMY_URL, jwt_audiences='["aud-a"]')

    def test_ipv6_bracket_literal_csv_string_is_still_accepted(self):
        """The detection must NOT misfire on the legit bracket-wrapped IPv6
        literal: ``[::1]:*`` carries no quote characters and is not
        ``{``-wrapped — the documented localhost default stays authorable."""
        settings = Settings(database_url=_DUMMY_URL, mcp_allowed_hosts="127.0.0.1:*,localhost:*,[::1]:*")
        assert settings.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]

    def test_empty_list_via_python_api_refuses_startup(self):
        """The empty-outage refusal is not bypassable through the Python API."""
        with pytest.raises(ValidationError, match="mcp_allowed_hosts must not be empty"):
            Settings(database_url=_DUMMY_URL, mcp_allowed_hosts=[])
