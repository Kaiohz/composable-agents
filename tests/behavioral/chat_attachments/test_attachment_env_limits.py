"""Behavioral tests — env-configured limit settings (REQ-092..096).

The five limits are read from environment variables at settings load time
(pydantic-settings) with spec defaults, silent fallback for absent/empty
values and HARD startup failure for present-but-invalid values, including
the coherence rule ``max_image_bytes <= max_document_bytes <= max_total_bytes``.

| Variable                        | Default     |
|---------------------------------|-------------|
| CHAT_ATTACHMENT_MAX_COUNT       | 10          |
| CHAT_ATTACHMENT_MAX_TOTAL_BYTES | 52428800    |
| CHAT_ATTACHMENT_MAX_IMAGE_BYTES | 10485760    |
| CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES | 52428800 |
| CREATE_PDF_MAX_MARKDOWN_CHARS   | 1000000     |
"""

import pytest
from pydantic import ValidationError

DEFAULTS = {
    "CHAT_ATTACHMENT_MAX_COUNT": "10",
    "CHAT_ATTACHMENT_MAX_TOTAL_BYTES": "52428800",
    "CHAT_ATTACHMENT_MAX_IMAGE_BYTES": "10485760",
    "CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES": "52428800",
    "CREATE_PDF_MAX_MARKDOWN_CHARS": "1000000",
}

FIELDS = {
    "CHAT_ATTACHMENT_MAX_COUNT": "chat_attachment_max_count",
    "CHAT_ATTACHMENT_MAX_TOTAL_BYTES": "chat_attachment_max_total_bytes",
    "CHAT_ATTACHMENT_MAX_IMAGE_BYTES": "chat_attachment_max_image_bytes",
    "CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES": "chat_attachment_max_document_bytes",
    "CREATE_PDF_MAX_MARKDOWN_CHARS": "create_pdf_max_markdown_chars",
}

ALL_VARS = set(DEFAULTS)


def _clean_env(env_dict, monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env_dict.items():
        monkeypatch.setenv(var, value)


def _make_settings(_env_overrides):
    """Instantiate the REAL Settings — env vars already staged by _clean_env."""
    from src.config import Settings

    return Settings(_env_file=None)


class TestChatAttachmentLimitSettings:
    def test_defaults_when_unset(self, monkeypatch):
        """REQ-094: absent variables silently use the spec defaults."""
        _clean_env({}, monkeypatch)
        settings = _make_settings({})
        for var, field in FIELDS.items():
            value = getattr(settings, field)
            assert value == int(DEFAULTS[var]), (var, value)

    def test_empty_values_fall_back_to_defaults(self, monkeypatch):
        """REQ-095: empty string values silently use the defaults."""
        _clean_env(dict.fromkeys(ALL_VARS, ""), monkeypatch)
        settings = _make_settings({})
        for var, field in FIELDS.items():
            value = getattr(settings, field)
            assert value == int(DEFAULTS[var]), (var, value)

    def test_env_override_honored(self, monkeypatch):
        overrides = dict(DEFAULTS)
        overrides["CHAT_ATTACHMENT_MAX_COUNT"] = "3"
        overrides["CREATE_PDF_MAX_MARKDOWN_CHARS"] = "5000"
        _clean_env(overrides, monkeypatch)
        settings = _make_settings({})
        assert settings.chat_attachment_max_count == 3
        assert settings.create_pdf_max_markdown_chars == 5000

    @pytest.mark.parametrize(
        "var,value",
        [
            ("CHAT_ATTACHMENT_MAX_COUNT", "not-a-number"),
            ("CHAT_ATTACHMENT_MAX_TOTAL_BYTES", "-1"),
            ("CHAT_ATTACHMENT_MAX_COUNT", "0"),
            ("CHAT_ATTACHMENT_MAX_IMAGE_BYTES", "abc"),
            ("CREATE_PDF_MAX_MARKDOWN_CHARS", ""),
        ],
    )
    def test_present_but_invalid_value_rejected(self, monkeypatch, var, value):
        """REQ-094/095: present + invalid blocks startup. NOTE: only values
        that are PRESENT but invalid block — an EMPTY string falls back to
        the default (covered above), so the empty case is expected to raise
        only if implementation reads it as strict int; per REQ-095 it must
        NOT raise. This entry pins invalid non-empty values."""
        if value == "":
            pytest.skip("empty values are covered by the fallback test above")
        overrides = dict(DEFAULTS)
        overrides[var] = value
        _clean_env(overrides, monkeypatch)
        with pytest.raises((ValidationError, ValueError)):
            _make_settings({})

    @pytest.mark.parametrize(
        "env_overrides",
        [
            # image > document → invalid hierarchy
            {
                "CHAT_ATTACHMENT_MAX_IMAGE_BYTES": "52428801",
                "CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES": "52428800",
            },
            # document > total → invalid hierarchy
            {
                "CHAT_ATTACHMENT_MAX_DOCUMENT_BYTES": "52428801",
                "CHAT_ATTACHMENT_MAX_TOTAL_BYTES": "52428800",
            },
        ],
    )
    def test_incoherent_hierarchy_rejected(self, monkeypatch, env_overrides):
        """REQ-094: max_image_bytes <= max_document_bytes <= max_total_bytes."""
        overrides = dict(DEFAULTS)
        overrides.update(env_overrides)
        _clean_env(overrides, monkeypatch)
        with pytest.raises((ValidationError, ValueError)):
            _make_settings({})

    def test_coherent_hierarchy_accepted(self, monkeypatch):
        overrides = dict(DEFAULTS)
        _clean_env(overrides, monkeypatch)
        settings = _make_settings({})
        assert (
            settings.chat_attachment_max_image_bytes
            <= settings.chat_attachment_max_document_bytes
            <= settings.chat_attachment_max_total_bytes
        )
