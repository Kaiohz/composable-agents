"""Behavioral tests — Model catalog reduced to model identity (spec §B
MODEL-001…MODEL-009 of model-catalog-settings-only.md).

Target rule: **Settings owns the provider connection (provider, base_url,
api_key); a Model resource owns ONLY its display name and the
``provider_model`` identifier to execute.**

Routes under test (real HTTP entry points on the real app):
* GET    /api/v1/models
* POST   /api/v1/models
* GET    /api/v1/models/{id}
* PATCH  /api/v1/models/{id}
* DELETE /api/v1/models/{id}

Contract under test (the Red target of this step):
* MODEL-001: ``ModelMetadata`` exposes exactly ``id, name, provider_model,
  created_at, updated_at`` — never ``provider`` nor ``base_url``.
* MODEL-002: POST accepts exactly ``name`` and ``provider_model`` (both
  required, non-empty after trim, existing length limits).
* MODEL-003: create/read/update/list responses must not contain ``provider``
  nor ``base_url``.
* MODEL-004: PATCH accepts a non-empty subset of ``name``/``provider_model``.
* MODEL-005: any request body containing ``provider``, ``base_url`` or any
  other unknown field is rejected with HTTP 422 (``extra="forbid"``).
* MODEL-006: an empty PATCH ``{}`` stays rejected; omitted fields keep their
  values.
* MODEL-008: ownership, non-enumeration and referenced-delete rules unchanged.
* MODEL-009: no credential field is ever accepted, stored or echoed.
"""

import uuid

import pytest

from tests.behavioral.agents.test_builtin_customization import (
    BUILTIN_NAME,
    built_in_entry,
    builtin_yaml_model_ref,
    put_override,
)
from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODE = 422
# The exactly-allowed identity field set (MODEL-001) — the legacy connection
# fields ``provider`` and ``base_url`` have been removed from the contract.
MODEL_FIELDS = {"id", "name", "provider_model", "created_at", "updated_at"}
# Fields removed from the Model contract (now owned exclusively by user LLM
# Settings) — no request may send them and no response may echo them.
REMOVED_FIELDS = ("provider", "base_url")


def model_payload(name="Primary model", provider_model="gpt-x"):
    return {"name": name, "provider_model": provider_model}


async def _create_model(client, headers, **overrides):
    payload = model_payload(**overrides)
    resp = await client.post("/api/v1/models", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def assert_identity_shape(body: dict) -> None:
    """MODEL-001/003: the exact identity field set, no removed fields."""
    assert set(body) == MODEL_FIELDS, (
        f"Model exposes unexpected fields: missing={MODEL_FIELDS - set(body)} extra={set(body) - MODEL_FIELDS}"
    )
    for removed in REMOVED_FIELDS:
        assert removed not in body, f"response must not contain the removed '{removed}' field"


class TestCreateModel:
    async def test_post_accepts_only_name_and_provider_model(self, client, users):
        """MODEL-002: the new contract requires ONLY name + provider_model."""
        resp = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json=model_payload(name="GLM 5.3 Flash", provider_model="z-ai/glm-5.3-flash:nitro"),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert_identity_shape(body)
        uuid.UUID(body["id"])
        assert body["name"] == "GLM 5.3 Flash"
        assert body["provider_model"] == "z-ai/glm-5.3-flash:nitro"

    async def test_legacy_connection_fields_are_rejected(self, client, users):
        """MODEL-005: ``provider`` and ``base_url`` no longer belong to a Model
        resource — the boundary must reject them with HTTP 422."""
        with_provider = model_payload() | {"provider": "openai"}
        rejected_provider = await client.post("/api/v1/models", headers=users["a"]["headers"], json=with_provider)
        assert rejected_provider.status_code == VALIDATION_CODE, rejected_provider.text

        with_base_url = model_payload() | {"base_url": "https://api.example.com/v1"}
        rejected_base_url = await client.post("/api/v1/models", headers=users["a"]["headers"], json=with_base_url)
        assert rejected_base_url.status_code == VALIDATION_CODE, rejected_base_url.text

        nothing_persisted = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert _items(nothing_persisted) == []

    async def test_required_fields_and_trim_rules(self, client, users):
        """MODEL-002: both values are required, non-empty after trim; values
        are trimmed by the existing validation behavior."""
        for payload in (
            model_payload(name=""),
            model_payload(provider_model=""),
            model_payload(name="   "),
            model_payload(provider_model="   "),
            {k: v for k, v in model_payload().items() if k != "provider_model"},
            {k: v for k, v in model_payload().items() if k != "name"},
        ):
            resp = await client.post("/api/v1/models", headers=users["a"]["headers"], json=payload)
            assert resp.status_code == VALIDATION_CODE, payload

        trimmed = await _create_model(client, users["a"]["headers"], name="  Padded  ", provider_model="  padded-x  ")
        assert trimmed["name"] == "Padded"
        assert trimmed["provider_model"] == "padded-x"

    async def test_no_credential_fields_accepted_or_leaked(self, client, users):
        """MODEL-009 / MDL-002 regression: credential material is still rejected
        outright at the boundary and never persisted nor echoed."""
        payload = model_payload()
        payload["api_key"] = "sk-should-never-be-stored"
        rejected = await client.post("/api/v1/models", headers=users["a"]["headers"], json=payload)
        assert rejected.status_code == VALIDATION_CODE, (
            f"credential fields must be rejected at the boundary, got {rejected.status_code}"
        )
        assert "sk-should-never-be-stored" not in rejected.text

        clean = await _create_model(client, users["a"]["headers"])
        assert_identity_shape(clean)
        assert "sk-should-never-be-stored" not in str(clean)

    async def test_unauthenticated_401(self, client):
        resp = await client.post("/api/v1/models", json=model_payload())
        assert resp.status_code == 401


class TestListAndGetModels:
    async def test_list_returns_owned_models_only_in_identity_shape(self, client, users):
        a = await _create_model(client, users["a"]["headers"])
        b = await _create_model(client, users["b"]["headers"])

        a_list = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert a_list.status_code == 200
        a_items = _items(a_list)
        for item in a_items:
            assert_identity_shape(item)
        assert a["id"] in [m["id"] for m in a_items]
        assert b["id"] not in [m["id"] for m in a_items]

        b_list = await client.get("/api/v1/models", headers=users["b"]["headers"])
        assert [m["id"] for m in _items(b_list)] == [b["id"]]

    async def test_get_owned_model_identity_shape(self, client, users):
        model = await _create_model(client, users["a"]["headers"], name="Readable", provider_model="read-x")
        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        body = got.json()
        assert_identity_shape(body)
        assert body["name"] == "Readable"
        assert body["provider_model"] == "read-x"

    async def test_get_unknown_404(self, client, users):
        resp = await client.get(f"/api/v1/models/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_get_foreign_model_404_non_enumerating(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.get(f"/api/v1/models/{model['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404
        assert model["name"] not in resp.text  # no metadata disclosure (SEC-001)


class TestPatchModel:
    async def test_patch_subset_keeps_omitted_field(self, client, users):
        """MODEL-006: a valid PATCH may update only one field; the omitted one
        keeps its current value."""
        model = await _create_model(client, users["a"]["headers"], name="Original", provider_model="orig-x")

        patched = await client.patch(
            f"/api/v1/models/{model['id']}",
            headers=users["a"]["headers"],
            json={"provider_model": "new-x"},
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["name"] == "Original"
        assert patched.json()["provider_model"] == "new-x"

        patched_name = await client.patch(
            f"/api/v1/models/{model['id']}",
            headers=users["a"]["headers"],
            json={"name": "Renamed"},
        )
        assert patched_name.status_code == 200
        assert patched_name.json()["name"] == "Renamed"
        assert patched_name.json()["provider_model"] == "new-x"

    async def test_patch_rejects_removed_and_unknown_fields(self, client, users):
        """MODEL-004/005: ``provider`` and ``base_url`` (and any other unknown
        field) are rejected with HTTP 422, no subset applied."""
        model = await _create_model(client, users["a"]["headers"], name="Stable", provider_model="stable-x")
        for payload in (
            {"provider": "anthropic"},
            {"base_url": "https://api.example.com/v1"},
            {"name": "Hijack", "provider": "openai"},
            {"unknown_field": "x"},
        ):
            resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json=payload)
            assert resp.status_code == VALIDATION_CODE, (payload, resp.text)

        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.json()["name"] == "Stable"

    async def test_patch_empty_object_rejected(self, client, users):
        """MODEL-006: an empty PATCH `{}` stays rejected."""
        model = await _create_model(client, users["a"]["headers"], name="Stable", provider_model="stable-x")
        resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={})
        assert resp.status_code == VALIDATION_CODE, resp.text

    async def test_patch_null_and_blank_fields_rejected(self, client, users):
        """MODEL-004: provided fields must be non-null and non-empty after trim."""
        model = await _create_model(client, users["a"]["headers"], name="Stable", provider_model="stable-x")
        for payload in (
            {"name": None},
            {"provider_model": None},
            {"name": ""},
            {"provider_model": "  "},
        ):
            resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json=payload)
            assert resp.status_code == VALIDATION_CODE, (payload, resp.text)
            assert "SELECT" not in resp.text and "asyncpg" not in resp.text  # no internal SQL echo
            got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
            assert got.status_code == 200
            field = next(iter(payload))
            assert got.json()[field] == model[field]

    async def test_patch_unknown_404(self, client, users):
        resp = await client.patch(f"/api/v1/models/{uuid.uuid4()}", headers=users["a"]["headers"], json={"name": "x"})
        assert resp.status_code == 404

    async def test_patch_foreign_model_404(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.patch(
            f"/api/v1/models/{model['id']}", headers=users["b"]["headers"], json={"name": "hijack"}
        )
        assert resp.status_code == 404


class TestDeleteModel:
    async def test_delete_unreferenced_204_then_get_404(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        deleted = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204
        gone = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert gone.status_code == 404

    async def test_delete_referenced_by_agent_returns_409(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        agent_name = f"agent-{uuid.uuid4().hex[:8]}"
        yaml_text = (
            f"name: {agent_name}\n"
            f"model_ref: {model['id']}\n"
            'system_prompt: "You are helpful."\n'
            "tools: []\n"
            "debug: false\n"
        )
        _, created = await create_agent_yaml(client, users["a"]["headers"], yaml_text, agent_name)
        assert created.status_code == 201

        deleted = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 409

    async def test_delete_foreign_model_404(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.delete(f"/api/v1/models/{model['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404

    async def test_delete_unknown_404(self, client, users):
        resp = await client.delete(f"/api/v1/models/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404


class TestMalformedNonUuidIds:
    """BUG-001 (MODEL-008/§9): a malformed (non-UUID) model id cannot exist —
    it must be answered with the SAME non-enumerating 404 as an unknown or
    foreign id, never an unhandled server error (500). The policy is enforced
    centrally in the shared repository path, so read, update and delete all
    inherit it, as does the MCP surface behind the same port."""

    async def test_get_malformed_id_is_a_clean_404_not_a_500(self, client, users):
        for model_id in ("not-a-uuid", "1234-5678", "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz"):
            resp = await client.get(f"/api/v1/models/{model_id}", headers=users["a"]["headers"])
            assert resp.status_code == 404, (model_id, resp.text)
            assert "Model not found" in resp.text  # same body as the unknown-id path

    async def test_patch_malformed_id_is_a_clean_404_not_a_500(self, client, users):
        resp = await client.patch("/api/v1/models/not-a-uuid", headers=users["a"]["headers"], json={"name": "x"})
        assert resp.status_code == 404, resp.text
        assert "Model not found" in resp.text

    async def test_delete_malformed_id_is_a_clean_404_not_a_500(self, client, users):
        resp = await client.delete("/api/v1/models/not-a-uuid", headers=users["a"]["headers"])
        assert resp.status_code == 404, resp.text
        assert "Model not found" in resp.text

    async def test_malformed_id_does_not_break_the_catalog(self, client, users):
        """The guard is id-parsing only: after a malformed id the route still
        serves real ids normally (no storage failure, no unrelated error)."""
        refused = await client.get("/api/v1/models/not-a-uuid", headers=users["a"]["headers"])
        assert refused.status_code == 404
        model = await _create_model(client, users["a"]["headers"])
        ok = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert ok.status_code == 200, ok.text


class TestDeleteModelReferencedByBuiltinCustomization:
    """Review C2 (MODEL-008): the built-in customization snapshot is NOT an
    ``agent_configs`` row — its ``model_ref`` is load-bearing for the listing
    projection, so a Model referenced ONLY there must produce the same in-use
    409 conflict as one referenced by a custom agent, while unreferenced
    Models stay deletable."""

    async def test_delete_referenced_by_builtin_customization_returns_409(self, client, users):
        """A Model referenced by the user's built-in customization is still in
        use (409); the listing keeps resolving."""
        model = await _create_model(
            client, users["a"]["headers"], name="Referenced", provider_model="builtin-referenced-x"
        )
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model["id"]))
        assert created.status_code == 200, created.text

        deleted = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 409, deleted.text

        # The owner's listing is untouched: the projected card still resolves.
        listing = await client.get("/api/v1/agents", headers=users["a"]["headers"])
        assert listing.status_code == 200, listing.text
        assert built_in_entry(listing.json())["model"] == "builtin-referenced-x"

    async def test_delete_unreferenced_model_with_a_customization_present_succeeds(self, client, users):
        """The guard must not over-block: a customization referencing a
        DIFFERENT Model leaves this unreferenced one deletable (204)."""
        referenced = await _create_model(client, users["a"]["headers"])
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=referenced["id"]))
        assert created.status_code == 200, created.text

        other = await _create_model(client, users["a"]["headers"], name="Deletable", provider_model="deletable-x")
        deleted = await client.delete(f"/api/v1/models/{other['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204, deleted.text

    async def test_another_users_customization_never_blocks_a_delete(self, client, users):
        """RLS scope: user B's built-in customization (referencing B's OWN
        model) must never conflict with user A deleting A's own unreferenced
        Model — the guard counts only the CALLER's snapshots."""
        model_b = await _create_model(client, users["b"]["headers"], name="B only", provider_model="fb-x")
        theirs = await put_override(client, users["b"]["headers"], builtin_yaml_model_ref(model_ref=model_b["id"]))
        assert theirs.status_code == 200, theirs.text

        model_a = await _create_model(client, users["a"]["headers"], name="A free", provider_model="fa-x")
        deleted = await client.delete(f"/api/v1/models/{model_a['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204, deleted.text

        # Sanity: user B still resolves their own customization.
        listing_b = await client.get("/api/v1/agents", headers=users["b"]["headers"])
        assert listing_b.status_code == 200
        assert built_in_entry(listing_b.json())["model"] == "fb-x"

    async def test_reset_of_the_customization_unblocks_the_delete(self, client, users):
        """Edge case (spec): resetting the built-in customization removes the
        reference — the previously blocked Model becomes deletable again."""
        model = await _create_model(client, users["a"]["headers"], name="Blocked", provider_model="blocked-x")
        created = await put_override(client, users["a"]["headers"], builtin_yaml_model_ref(model_ref=model["id"]))
        assert created.status_code == 200, created.text
        blocked = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert blocked.status_code == 409, blocked.text

        reset = await client.delete(f"/api/v1/agents/{BUILTIN_NAME}/customization", headers=users["a"]["headers"])
        assert reset.status_code == 204, reset.text

        deleted = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204, deleted.text

    async def test_corrupt_override_snapshot_does_not_block_a_delete(self, client, users, stack):
        """A corrupt (unparsable) snapshot is skipped by the in-use guard: it
        must not block the delete — its own resolution error remains the
        read-time behavior of the listing, never a write-time blocker."""
        from sqlalchemy import text

        from src.infrastructure.database.rls_context import system_rls_context

        async with system_rls_context(), stack["engine"].begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO builtin_agent_overrides "
                    "(user_id, agent_name, yaml, created_at, updated_at) "
                    "VALUES (:user_id, :agent_name, :yaml, now(), now())"
                ),
                {"user_id": users["a"]["id"], "agent_name": BUILTIN_NAME, "yaml": "not: [valid"},
            )

        model = await _create_model(client, users["a"]["headers"], name="Deletable anyway", provider_model="ok-x")
        deleted = await client.delete(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert deleted.status_code == 204, deleted.text


def _items(response):
    body = response.json()
    if isinstance(body, dict) and "items" in body:
        return body["items"]
    assert isinstance(body, list), f"Unexpected list-response shape: {body!r}"
    return body
