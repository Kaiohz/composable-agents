"""Behavioral tests — metadata-only model catalog (spec §5.2 MDL-001…MDL-004).

Routes under test (real HTTP entry points on the real app):
* GET    /api/v1/models
* POST   /api/v1/models
* GET    /api/v1/models/{id}
* PATCH  /api/v1/models/{id}
* DELETE /api/v1/models/{id}

MDL-002 is enforced strictly: responses carry ONLY the feature-owned metadata
fields — no credential字段 of any kind (spec "Models do not contain API keys").
"""

import uuid

import pytest

from tests.behavioral.conftest import create_agent_yaml

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}
# The exactly-allowed metadata field set (MDL-001):
MODEL_FIELDS = {"id", "name", "provider", "provider_model", "base_url", "created_at", "updated_at"}


def model_payload(name="Primary model", provider="openai", provider_model="gpt-x", base_url=None):
    body = {"name": name, "provider": provider, "provider_model": provider_model}
    if base_url is not None:
        body["base_url"] = base_url
    return body


async def _create_model(client, headers, **overrides):
    payload = model_payload(**overrides)
    resp = await client.post("/api/v1/models", headers=headers, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestCreateModel:
    async def test_post_returns_201_with_metadata_only_shape(self, client, users):
        resp = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json=model_payload(base_url="https://api.example.com/v1"),
        )
        assert resp.status_code == 201
        body = resp.json()
        assert set(body) == MODEL_FIELDS, f"Model exposes unexpected fields: {set(body) - MODEL_FIELDS}"
        uuid.UUID(body["id"])
        assert body["name"] == "Primary model"
        assert body["provider"] == "openai"
        assert body["provider_model"] == "gpt-x"
        assert body["base_url"] == "https://api.example.com/v1"

    async def test_base_url_optional_but_validated(self, client, users):
        ok = await client.post("/api/v1/models", headers=users["a"]["headers"], json=model_payload(base_url=None))
        assert ok.status_code == 201
        assert ok.json()["base_url"] is None

        bad = await client.post(
            "/api/v1/models", headers=users["a"]["headers"], json=model_payload(base_url="not-a-url")
        )
        assert bad.status_code in VALIDATION_CODES

    async def test_required_fields_validated(self, client, users):
        cases = [
            model_payload(name=""),
            model_payload(provider=""),
            model_payload(provider_model=""),
            {k: v for k, v in model_payload().items() if k != "provider"},
            {k: v for k, v in model_payload().items() if k != "provider_model"},
            {k: v for k, v in model_payload().items() if k != "name"},
        ]
        for payload in cases:
            resp = await client.post("/api/v1/models", headers=users["a"]["headers"], json=payload)
            assert resp.status_code in VALIDATION_CODES, payload

    async def test_no_credential_fields_accepted_or_leaked(self, client, users):
        """MDL-002: models must not contain API keys/JWTs — attempts to send
        credential fields are REJECTED outright (DTO ``extra="forbid"``, QA
        BUG-009 boundary hardening), so they are neither persisted nor echoed."""
        payload = model_payload(base_url="https://api.example.com/v1")
        payload["api_key"] = "sk-should-never-be-stored"
        rejected = await client.post("/api/v1/models", headers=users["a"]["headers"], json=payload)
        assert rejected.status_code in {400, 422}, (
            f"credential fields must be rejected at the boundary, got {rejected.status_code}"
        )
        assert "sk-should-never-be-stored" not in rejected.text

        # A clean payload (without the credential field) succeeds and still
        # carries/echoes no credential data.
        clean = await client.post(
            "/api/v1/models",
            headers=users["a"]["headers"],
            json=model_payload(base_url="https://api.example.com/v1"),
        )
        assert clean.status_code == 201
        body = clean.json()
        assert set(body) == MODEL_FIELDS
        assert "sk-should-never-be-stored" not in clean.text

    async def test_unauthenticated_401(self, client):
        resp = await client.post("/api/v1/models", json=model_payload())
        assert resp.status_code == 401


class TestListAndGetModels:
    async def test_list_returns_owned_models_only(self, client, users):
        a = await _create_model(client, users["a"]["headers"])
        b = await _create_model(client, users["b"]["headers"])

        a_list = await client.get("/api/v1/models", headers=users["a"]["headers"])
        assert a_list.status_code == 200
        a_ids = [m["id"] for m in _items(a_list)]
        assert a["id"] in a_ids
        assert b["id"] not in a_ids

        b_list = await client.get("/api/v1/models", headers=users["b"]["headers"])
        assert [m["id"] for m in _items(b_list)] == [b["id"]]

    async def test_get_owned_model(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        got = await client.get(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"])
        assert got.status_code == 200
        assert set(got.json()) == MODEL_FIELDS

    async def test_get_unknown_404(self, client, users):
        resp = await client.get(f"/api/v1/models/{uuid.uuid4()}", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_get_foreign_model_404_non_enumerating(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.get(f"/api/v1/models/{model['id']}", headers=users["b"]["headers"])
        assert resp.status_code == 404
        assert model["name"] not in resp.text  # no metadata disclosure (SEC-001)


class TestPatchModel:
    async def test_patch_updates_all_editable_fields(self, client, users):
        model = await _create_model(
            client, users["a"]["headers"], provider="openai", provider_model="gpt-a"
        )
        patched = await client.patch(
            f"/api/v1/models/{model['id']}",
            headers=users["a"]["headers"],
            json={"name": "Edited", "provider": "anthropic", "provider_model": "claude-b", "base_url": None},
        )
        assert patched.status_code == 200
        body = patched.json()
        assert body["name"] == "Edited"
        assert body["provider"] == "anthropic"
        assert body["provider_model"] == "claude-b"
        assert body["base_url"] is None

    async def test_patch_validates_fields(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        for payload in ({"provider": ""}, {"provider_model": ""}, {"name": ""}, {"base_url": "nope"}):
            resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json=payload)
            assert resp.status_code in VALIDATION_CODES, payload

    async def test_patch_unknown_404(self, client, users):
        resp = await client.patch(f"/api/v1/models/{uuid.uuid4()}", headers=users["a"]["headers"], json={"name": "x"})
        assert resp.status_code == 404

    async def test_patch_foreign_model_404(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["b"]["headers"], json={"name": "hijack"})
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
            f"model: legacy-string\n"
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


def _items(response):
    body = response.json()
    if isinstance(body, dict) and "items" in body:
        return body["items"]
    assert isinstance(body, list), f"Unexpected list-response shape: {body!r}"
    return body


class TestExplicitNullNonNullableBoundary:
    """QA PASS-3 (BUG-009): an explicit ``null`` for the non-nullable model
    fields is a clean 4xx validation error at the request boundary — never a
    5xx with internal SQL echoed — while the spec-declared nullable
    ``base_url`` remains clearable (the 200 control case)."""

    async def test_patch_null_name_is_a_client_error(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={"name": None})
        assert resp.status_code in {400, 422}, f"null name must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_provider_is_a_client_error(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.patch(f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={"provider": None})
        assert resp.status_code in {400, 422}, f"null provider must be 4xx, got {resp.status_code} {resp.text}"

    async def test_patch_null_provider_model_is_a_client_error(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        resp = await client.patch(
            f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={"provider_model": None}
        )
        assert resp.status_code in {400, 422}, f"null provider_model must be 4xx, got {resp.status_code} {resp.text}"

    async def test_null_patches_never_echo_internal_sql(self, client, users):
        model = await _create_model(client, users["a"]["headers"])
        mid = model["id"]
        for field in ("name", "provider", "provider_model"):
            resp = await client.patch(f"/api/v1/models/{mid}", headers=users["a"]["headers"], json={field: None})
            body = resp.text
            assert "SELECT" not in body and "asyncpg" not in body and "NOT NULL" not in body.upper(), body
            # The stored row must be untouched after each rejected patch.
            got = await client.get(f"/api/v1/models/{mid}", headers=users["a"]["headers"])
            assert got.status_code == 200
            assert got.json()[field] == model[field]

    async def test_patch_base_url_null_still_clears_the_nullable_field(self, client, users):
        model = await _create_model(client, users["a"]["headers"], base_url="https://api.example.com/v1")
        resp = await client.patch(
            f"/api/v1/models/{model['id']}", headers=users["a"]["headers"], json={"base_url": None}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["base_url"] is None
