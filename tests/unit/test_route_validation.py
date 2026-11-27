"""Unit tests for the shared route-boundary validation (REQ-007 / REQ-013).

Locks the store-path contract shared by REST and MCP: plain component names
are accepted; traversal, separators and empties are refused with the
centralized store-file error.
"""

import pytest

from src.application.services.route_validation import (
    memory_store_path,
    normalize_store_path,
    read_yaml_bytes,
    skill_directory_path,
    skill_store_path,
)
from src.domain.errors.config import ConfigError
from src.domain.errors.store_file import StoreFileNotFoundError

_VALID_NAMES = ["my-skill", "rag.summarize", "notes_2026", "a", "x y".replace(" ", "-")]
# Values a store component must REFUSE (traversal, separators, empty). Note:
# lone "-" and slash-wrapped names ("/abs/") are LEGAL components — stripping
# the wrapping slashes is the established normalization.
_UNSAFE_COMPONENTS = ["../escape", "../../etc/passwd", "a/b", "a\\b", "..", ".", "", "  ", "a/../b"]


class TestRejectUnsafeComponent:
    @pytest.mark.parametrize("name", ["my-skill", "rag.tools", "a"])
    def test_component_passes_through_stripped(self, name):
        from src.application.services.route_validation import _reject_unsafe_component

        assert _reject_unsafe_component(name, "invalid") == name
        assert _reject_unsafe_component(f"/{name}/", "invalid") == name

    @pytest.mark.parametrize("name", _UNSAFE_COMPONENTS)
    def test_traversal_and_separators_are_refused(self, name):
        from src.application.services.route_validation import _reject_unsafe_component

        with pytest.raises(StoreFileNotFoundError):
            _reject_unsafe_component(name, "invalid")


class TestSkillStorePath:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("my-skill", "/skills/my-skill/SKILL.md"), ("rag.tools", "/skills/rag.tools/SKILL.md")],
    )
    def test_resolves_under_the_global_prefix(self, name, expected):
        assert skill_store_path(name) == expected

    @pytest.mark.parametrize("name", _UNSAFE_COMPONENTS)
    def test_traversal_is_refused(self, name):
        with pytest.raises(StoreFileNotFoundError):
            skill_store_path(name)

    def test_filename_is_appended_verbatim(self):
        assert skill_store_path("my-skill", "SKILL.md") == "/skills/my-skill/SKILL.md"


class TestSkillDirectoryPath:
    def test_resolves_the_directory_prefix(self):
        assert skill_directory_path("my-skill") == "/skills/my-skill/"

    @pytest.mark.parametrize("name", _UNSAFE_COMPONENTS)
    def test_traversal_is_refused(self, name):
        with pytest.raises(StoreFileNotFoundError):
            skill_directory_path(name)


class TestMemoryStorePath:
    # REQ-015: ``a/b`` is NO LONGER unsafe for memories — it is the accepted
    # nested agent shape (agent segment validated). The remaining traversal
    # components stay refused for the personal namespace.
    _UNSAFE_MEMORY_COMPONENTS = [c for c in _UNSAFE_COMPONENTS if c != "a/b"]

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [("AGENTS.md", "/memories/AGENTS.md"), ("notes.md", "/memories/notes.md")],
    )
    def test_resolves_under_the_global_prefix(self, filename, expected):
        assert memory_store_path(filename) == expected

    @pytest.mark.parametrize("filename", _UNSAFE_MEMORY_COMPONENTS)
    def test_prefix_escape_is_refused(self, filename):
        with pytest.raises(StoreFileNotFoundError):
            memory_store_path(filename)

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [("agent-x/notes.md", "/memories/agent-x/notes.md"), ("a0/b.md", "/memories/a0/b.md")],
    )
    def test_nested_agent_shape_is_accepted(self, filename, expected):
        """REQ-015: the nested shape ``{agent_slug}/{name}`` resolves to
        ``/memories/{agent_slug}/{name}`` when the agent segment matches the
        REQ-002 charset."""
        assert memory_store_path(filename) == expected

    @pytest.mark.parametrize(
        "filename",
        ["bad_agent/notes.md", "BadAgent/notes.md", "agent/deep/notes.md", "../escape-x/notes.md"],
    )
    def test_invalid_agent_segments_are_refused(self, filename):
        """REQ-015: invalid agent segments and deeper nesting stay refused."""
        with pytest.raises(StoreFileNotFoundError):
            memory_store_path(filename)

    def test_bound_scope_resolves_inside_the_agent_space(self):
        """REQ-020: with a bound agent_scope the filename must be a plain
        component and resolves inside ``/memories/{agent_scope}/``."""
        assert memory_store_path("notes.md", agent_scope="agent-x") == "/memories/agent-x/notes.md"
        with pytest.raises(StoreFileNotFoundError):
            memory_store_path("a/b.md", agent_scope="agent-x")

    def test_bound_scope_cannot_bind_the_reserved_personal_namespace(self):
        """Review N-1 layer (d): a bound agent scope named 'personal' maps
        onto its own distinct fallback slug — never the flat personal
        namespace (which belongs to scope-less sessions)."""
        from src.domain.services.agent_scope import agent_scope_slug

        bound = agent_scope_slug("personal")
        assert bound != "personal"
        assert memory_store_path("notes.md", agent_scope="personal") == f"/memories/{bound}/notes.md"
        assert (
            memory_store_path("notes.md", agent_scope="Personal")
            == f"/memories/{agent_scope_slug('Personal')}/notes.md"
        )
        assert memory_store_path("notes.md", agent_scope="personal") != "/memories/personal/notes.md"


class TestNormalizeStorePath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [("a/b", "/a/b"), ("/a/b", "/a/b"), ("a", "/a")],
    )
    def test_leading_slash_is_added(self, path, expected):
        assert normalize_store_path(path) == expected

    @pytest.mark.parametrize("path", ["../escape", "/a/../b", "a/.."])
    def test_traversal_is_refused(self, path):
        with pytest.raises(StoreFileNotFoundError):
            normalize_store_path(path)


class TestReadYamlBytes:
    def test_valid_utf8_decodes(self):
        assert read_yaml_bytes(b"name: a\n") == "name: a\n"

    def test_oversized_payload_is_refused(self):
        from src.application.services.route_validation import MAX_YAML_SIZE

        with pytest.raises(ConfigError, match="too large"):
            read_yaml_bytes(b"x" * (MAX_YAML_SIZE + 1))

    def test_non_utf8_is_refused(self):
        with pytest.raises(ConfigError, match="UTF-8"):
            read_yaml_bytes(b"\xff\xfe\x00binary")
