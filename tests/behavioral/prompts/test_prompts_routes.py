"""Behavioral tests — Local prompt registry CRUD (spec §5.1 PR-001…PR-005).

Routes under test (real HTTP entry points on the real app):
* POST   /api/v1/prompts
* GET    /api/v1/prompts
* GET    /api/v1/prompts/{id}
* PATCH  /api/v1/prompts/{id}
* DELETE /api/v1/prompts/{id}

The Phoenix endpoint is deliberately unreachable (conftest env) so every test
proves that no Phoenix call backs the local registry (spec §8.8). Status codes
left unpinned by the spec use the documented tolerance constants below.
"""

import uuid

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Status-code families the spec leaves unpinned:
VALIDATION_CODES = {400, 422}  # request/domain validation failures


def prompt_payload(name="Research prompt", description=None, content="You are a research agent."):
    body: dict = {"name": name, "content": content}
    if description is not None:
        body["description"] = description
    return body


class TestCreatePrompt:
    async def test_post_returns_201_with_metadata_and_version_1(self, client, users):
        resp = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        assert resp.status_code == 201
        body = resp.json()
        # UUID identity (PR-001: identity is the UUID, not the name)
        uuid.UUID(body["id"])
        assert body["name"] == "Research prompt"
        assert body["description"] is None
        # Creating atomically creates immutable version 1 (PR-003)
        assert body["latest_version"] == 1
        assert body["created_at"] is not None
        assert body["updated_at"] is not None

    async def test_created_prompt_is_readable_and_has_version_row(self, client, users):
        create = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        prompt_id = create.json()["id"]

        got = await client.get(f"/api/v1/prompts/{prompt_id}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert got.json()["id"] == prompt_id

        versions = await client.get(f"/api/v1/prompts/{prompt_id}/versions", headers=users["a"]["headers"])
        assert versions.status_code == 200
        version_numbers = [v["version"] for v in _items(versions)]
        assert version_numbers == [1]

    async def test_name_is_trimmed_and_empty_name_rejected(self, client, users):
        # Trimmed non-empty (PR-002)
        ok = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(name="  spaced name  ")
        )
        assert ok.status_code == 201
        assert ok.json()["name"] == "spaced name"

        # Empty / whitespace-only names are rejected
        for bad in ("", "   "):
            resp = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(name=bad))
            assert resp.status_code in VALIDATION_CODES

    async def test_missing_name_rejected(self, client, users):
        resp = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json={"description": "no name", "content": "c"}
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_content_is_required(self, client, users):
        resp = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json={"name": "no content"})
        assert resp.status_code in VALIDATION_CODES

    async def test_content_may_be_empty_string_and_stored_verbatim(self, client, users):
        verbatim = "  keep \n every \t whitespace  "
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(content=""))
        assert created.status_code == 201

        verbatim_created = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(name="verbatim", content=verbatim)
        )
        assert verbatim_created.status_code == 201
        vid = verbatim_created.json()["id"]

        version = await client.get(f"/api/v1/prompts/{vid}/versions/1", headers=users["a"]["headers"])
        assert version.status_code == 200
        # Stored and returned verbatim, including whitespace (PR-002)
        assert version.json()["content"] == verbatim

    async def test_duplicate_display_names_permitted_with_distinct_ids(self, client, users):
        first = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        second = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["id"] != second.json()["id"]


class TestListPrompts:
    async def test_get_list_returns_owned_metadata(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        listing = await client.get("/api/v1/prompts", headers=users["a"]["headers"])
        assert listing.status_code == 200
        ids = [item["id"] for item in _items(listing)]
        assert created.json()["id"] in ids
        for item in _items(listing):
            assert {"id", "name", "latest_version"} <= set(item)

    async def test_list_is_scoped_to_owner(self, client, users):
        a = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        b_list = await client.get("/api/v1/prompts", headers=users["b"]["headers"])
        assert b_list.status_code == 200
        assert a.json()["id"] not in [item["id"] for item in _items(b_list)]


class TestGetPrompt:
    async def test_get_returns_owned_prompt(self, client, users):
        created = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(description="d")
        )
        got = await client.get(f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        body = got.json()
        assert body["description"] == "d"
        assert body["latest_version"] == 1

    async def test_get_unknown_id_404_non_enumerating(self, client, users):
        resp = await client.get(f"/api/v1/prompts/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_get_other_users_prompt_404_non_enumerating(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.get(f"/api/v1/prompts/{created.json()['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404
        # No metadata disclosure to the foreign user (SEC-001)
        assert created.json()["name"] not in resp.text

    async def test_unauthenticated_request_401(self, client):
        resp = await client.get("/api/v1/prompts")
        assert resp.status_code == 401


class TestPatchPrompt:
    async def test_metadata_edit_updates_without_creating_version(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        pid = created.json()["id"]

        patched = await client.patch(
            f"/api/v1/prompts/{pid}",
            headers=users["a"]["headers"],
            json={"name": "Renamed", "description": "Better description"},
        )
        assert patched.status_code == 200
        assert patched.json()["name"] == "Renamed"
        # Metadata-only edits do not create a version (PR-003)
        assert patched.json()["latest_version"] == 1

        versions = await client.get(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"])
        assert len(_items(versions)) == 1

    async def test_patch_rejects_content(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.patch(
            f"/api/v1/prompts/{created.json()['id']}",
            headers=users["a"]["headers"],
            json={"content": "content is not accepted here"},
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_patch_empty_name_rejected(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.patch(
            f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"], json={"name": " "}
        )
        assert resp.status_code in VALIDATION_CODES

    async def test_patch_unknown_id_404(self, client, users):
        resp = await client.patch(f"/api/v1/prompts/{uuid.uuid4()}", headers=users["a"]["headers"], json={"name": "x"})
        assert resp.status_code == 404

    async def test_patch_other_users_prompt_404(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.patch(
            f"/api/v1/prompts/{created.json()['id']}", headers=users["b"]["headers"], json={"name": "hijack"}
        )
        assert resp.status_code == 404

    async def test_unauthenticated_patch_401(self, client):
        resp = await client.patch(f"/api/v1/prompts/{uuid.uuid4()}", json={"name": "x"})
        assert resp.status_code == 401


class TestDeletePrompt:
    async def test_delete_unreferenced_prompt_204(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        deleted = await client.delete(f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204
        gone = await client.get(f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"])
        assert gone.status_code == 404

    async def test_delete_other_users_prompt_404(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.delete(f"/api/v1/prompts/{created.json()['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404

    async def test_delete_unknown_id_404(self, client, users):
        resp = await client.delete(f"/api/v1/prompts/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404


class TestPatchDescriptionClearing:
    """QA BUG-002: an explicit ``"description": null`` clears the description."""

    async def test_patch_description_null_clears_it(self, client, users):
        created = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": "QA clearing prompt", "description": "to be cleared", "content": "c"},
        )
        assert created.status_code == 201
        pid = created.json()["id"]

        cleared = await client.patch(
            f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"description": None}
        )
        assert cleared.status_code == 200, cleared.text
        body = cleared.json()
        assert body["description"] is None
        # Metadata-only edit: no version is created (PR-003).
        assert body["latest_version"] == 1

    async def test_patch_name_only_keeps_cleared_description(self, client, users):
        created = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": "QA clearing prompt", "description": "d1", "content": "c"},
        )
        pid = created.json()["id"]
        cleared = await client.patch(
            f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"description": None}
        )
        assert cleared.status_code == 200

        renamed = await client.patch(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"name": "QA v2"})
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "QA v2"
        assert renamed.json()["description"] is None

    async def test_patch_empty_body_rejected(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        resp = await client.patch(f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"], json={})
        assert resp.status_code in {400, 422}


class TestResponseFieldSets:
    """QA BUG-005: HTTP payloads must not carry the owner id."""

    async def test_prompt_create_payload_has_no_user_id(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        assert created.status_code == 201
        assert "user_id" not in created.json(), f"owner id leaked in prompt payload: {created.json()}"

    async def test_prompt_get_payload_has_no_user_id(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        got = await client.get(f"/api/v1/prompts/{created.json()['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert "user_id" not in got.json(), f"owner id leaked in prompt payload: {got.json()}"

    async def test_prompt_list_payloads_have_no_user_id(self, client, users):
        await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        listing = await client.get("/api/v1/prompts", headers=users["a"]["headers"])
        assert listing.status_code == 200
        for item in _items(listing):
            assert "user_id" not in item, f"owner id leaked in prompt list payload: {item}"


class TestVersionListShape:
    """QA BUG-004: the version list carries number + timestamp only."""

    async def test_version_list_payloads_have_no_content_or_user_id(self, client, users):
        created = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload(content="full v1")
        )
        pid = created.json()["id"]
        await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "full v2"})

        history = await client.get(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"])
        assert history.status_code == 200
        items = history.json() if isinstance(history.json(), list) else history.json()["items"]
        assert [item["version"] for item in items] == [1, 2]
        for item in items:
            assert "content" not in item, f"version list leaked content: {item}"
            assert "user_id" not in item, f"version list leaked the owner id: {item}"
            assert item["created_at"] is not None

    async def test_version_create_payload_has_no_user_id(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        pid = created.json()["id"]
        new_version = await client.post(
            f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v2"}
        )
        assert new_version.status_code == 201
        assert "user_id" not in new_version.json(), f"owner id leaked in version payload: {new_version.json()}"
        # The immutable content stays reachable through the per-version GET.
        got = await client.get(
            f"/api/v1/prompts/{pid}/versions/{new_version.json()['version']}", headers=users["a"]["headers"]
        )
        assert got.status_code == 200
        assert got.json()["content"] == "v2"
        assert "user_id" not in got.json()


class TestPhoenixIndependence:
    async def test_prompt_crud_works_with_phoenix_unreachable(self, client, users):
        """PR-001: no prompt operation may call Phoenix — with Phoenix dead,
        create/read/delete still work end-to-end (conftest env points
        TRACING__PHOENIX_COLLECTOR_ENDPOINT at an unroutable port)."""
        resp = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        assert resp.status_code == 201
        pid = resp.json()["id"]

        listing = await client.get("/api/v1/prompts", headers=users["a"]["headers"])
        assert listing.status_code == 200

        version = await client.post(
            f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v2 without phoenix"}
        )
        assert version.status_code == 201

        deleted = await client.delete(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"])
        assert deleted.status_code == 204


def _items(response):
    """Extract the item list from a list response ({"items": [...]} or bare list)."""
    body = response.json()
    if isinstance(body, dict) and "items" in body:
        return body["items"]
    assert isinstance(body, list), f"Unexpected list-response shape: {body!r}"
    return body


class TestExplicitNullNonNullableBoundary:
    """QA PASS-3 (BUG-006): an explicit ``null`` for the NON-nullable prompt
    field (``name``) is a clean 4xx validation error at the request boundary —
    never a 5xx with internal SQL echoed — while the spec-declared nullable
    ``description`` remains clearable."""

    async def test_patch_name_null_is_a_client_error(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        pid = created.json()["id"]
        resp = await client.patch(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"name": None})
        assert resp.status_code in {400, 422}, f"null name must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_name_null_does_not_echo_internals(self, client, users):
        created = await client.post("/api/v1/prompts", headers=users["a"]["headers"], json=prompt_payload())
        pid = created.json()["id"]
        resp = await client.patch(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"name": None})
        body = resp.text
        assert "SELECT" not in body and "prompts" not in body.lower().replace("prompts/{id}", ""), body
        assert "asyncpg" not in body and "NOT NULL" not in body.upper(), body

    async def test_patch_name_null_leaves_metadata_untouched(self, client, users):
        created = await client.post(
            "/api/v1/prompts",
            headers=users["a"]["headers"],
            json={"name": "keep me", "description": "d", "content": "c"},
        )
        pid = created.json()["id"]
        await client.patch(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"name": None})
        got = await client.get(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"])
        assert got.status_code == 200
        body = got.json()
        assert body["name"] == "keep me", "a rejected PATCH must leave the stored name untouched"
        assert body["description"] == "d"

    async def test_patch_description_clearing_still_works(self, client, users):
        """Regression guard: the BUG-002 semantics stay intact."""
        created = await client.post(
            "/api/v1/prompts", headers=users["a"]["headers"], json={"name": "n", "description": "d", "content": "c"}
        )
        pid = created.json()["id"]
        resp = await client.patch(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"], json={"description": None})
        assert resp.status_code == 200
        assert resp.json()["description"] is None
