"""Behavioral tests — immutable prompt versions (spec §5.1 PR-002, PR-003).

Routes under test:
* GET  /api/v1/prompts/{id}/versions
* POST /api/v1/prompts/{id}/versions
* GET  /api/v1/prompts/{id}/versions/{version}

Version allocation must be transactional and safe under concurrent
submissions (real PostgreSQL, real HTTP concurrency) — PR-003 and success
criterion 2.3.
"""

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

VALIDATION_CODES = {400, 422}


async def _create_prompt(client, headers, name="Versions prompt", content="v1 content"):
    resp = await client.post("/api/v1/prompts", headers=headers, json={"name": name, "content": content})
    assert resp.status_code == 201
    return resp.json()


class TestVersionCreation:
    async def test_second_content_creates_version_2_and_bumps_latest(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]

        new_version = await client.post(
            f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v2 content"}
        )
        assert new_version.status_code == 201
        assert new_version.json()["version"] == 2

        prompt_after = await client.get(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"])
        assert prompt_after.json()["latest_version"] == 2  # updates latest_version (PR-003)

    async def test_content_required_and_versions_positive(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]

        resp = await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={})
        assert resp.status_code in VALIDATION_CODES

    async def test_version_content_stored_verbatim(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]
        verbatim = "\n  tricky \n content with trailing spaces   "
        created = await client.post(
            f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": verbatim}
        )
        assert created.status_code == 201
        got = await client.get(
            f"/api/v1/prompts/{pid}/versions/{created.json()['version']}", headers=users["a"]["headers"]
        )
        assert got.status_code == 200
        assert got.json()["content"] == verbatim

    async def test_versions_sequentially_numbered(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]
        for expected in (2, 3, 4):
            resp = await client.post(
                f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": f"v{expected}"}
            )
            assert resp.status_code == 201
            assert resp.json()["version"] == expected

    async def test_unknown_prompt_returns_404(self, client, users):
        resp = await client.post(
            f"/api/v1/prompts/{uuid.uuid4()}/versions", headers=users["a"]["headers"], json={"content": "x"}
        )
        assert resp.status_code == 404

    async def test_other_users_prompt_returns_404(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        resp = await client.post(
            f"/api/v1/prompts/{prompt['id']}/versions", headers=users["b"]["headers"], json={"content": "evil"}
        )
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        resp = await client.post(f"/api/v1/prompts/{prompt['id']}/versions", json={"content": "x"})
        assert resp.status_code == 401


class TestVersionList:
    async def test_history_lists_versions_ascending_with_timestamps_only(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]
        await client.post(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "v2"})

        history = await client.get(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"])
        assert history.status_code == 200
        items = history.json() if isinstance(history.json(), list) else history.json()["items"]
        assert [item["version"] for item in items] == [1, 2]  # numerically ordered (PR-003)
        for item in items:
            assert item["created_at"] is not None

    async def test_history_empty_for_unknown_prompt(self, client, users):
        resp = await client.get(f"/api/v1/prompts/{uuid.uuid4()}/versions", headers=users["a"]["headers"])
        assert resp.status_code == 404


class TestVersionRead:
    async def test_pinned_read_returns_exact_immutable_content(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"], content="immutable v1")
        pid = prompt["id"]
        await client.post(
            f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"], json={"content": "mutable v2"}
        )

        v1 = await client.get(f"/api/v1/prompts/{pid}/versions/1", headers=users["a"]["headers"])
        assert v1.status_code == 200
        assert v1.json()["content"] == "immutable v1"  # older version unchanged — immutability (PR-003)

        v2 = await client.get(f"/api/v1/prompts/{pid}/versions/2", headers=users["a"]["headers"])
        assert v2.json()["content"] == "mutable v2"

    async def test_nonexistent_version_404(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        resp = await client.get(f"/api/v1/prompts/{prompt['id']}/versions/99", headers=users["a"]["headers"])
        assert resp.status_code == 404

    async def test_zero_or_negative_version_404_or_422(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        for bad in ("0", "-1"):
            resp = await client.get(f"/api/v1/prompts/{prompt['id']}/versions/{bad}", headers=users["a"]["headers"])
            assert resp.status_code in {404, *VALIDATION_CODES}

    async def test_other_users_version_404(self, client, users):
        prompt = await _create_prompt(client, users["a"]["headers"])
        resp = await client.get(f"/api/v1/prompts/{prompt['id']}/versions/1", headers=users["b"]["headers"])
        assert resp.status_code == 404


class TestConcurrentVersionAllocation:
    async def test_concurrent_submissions_allocate_distinct_versions(self, client, client_factory, users):
        """Success criterion 2.3: concurrent content submissions never reuse a
        version number nor overwrite another version (real PostgreSQL)."""
        prompt = await _create_prompt(client, users["a"]["headers"])
        pid = prompt["id"]

        async def submit(index: int):
            ac = await client_factory()
            try:
                return await ac.post(
                    f"/api/v1/prompts/{pid}/versions",
                    headers=users["a"]["headers"],
                    json={"content": f"concurrent v{index}"},
                )
            finally:
                await ac.aclose()

        results = await asyncio.gather(submit(1), submit(2), submit(3))

        # Every submission either succeeds with a distinct version or fails
        # with an explicit conflict — never a duplicate version number.
        versions = [r.json()["version"] for r in results if r.status_code == 201]
        conflict_statuses = [r.status_code for r in results if r.status_code != 201]
        assert all(status == 409 for status in conflict_statuses), conflict_statuses
        assert len(versions) == len(set(versions)), f"Duplicate versions allocated: {versions}"
        assert all(v >= 2 for v in versions)

        latest = await client.get(f"/api/v1/prompts/{pid}", headers=users["a"]["headers"])
        assert latest.json()["latest_version"] == max(versions)

        history = await client.get(f"/api/v1/prompts/{pid}/versions", headers=users["a"]["headers"])
        items = history.json() if isinstance(history.json(), list) else history.json()["items"]
        numbers = [item["version"] for item in items]
        assert len(numbers) == len(set(numbers)), "Duplicate version rows persisted"
        assert numbers == sorted(numbers)

    async def test_concurrent_creates_never_overwrite_versions(self, client, client_factory, users):
        """Two concurrent creates with the same name each get their own
        prompt + version 1 (duplicate display names are permitted)."""

        async def create(index: int):
            ac = await client_factory()
            try:
                return await ac.post(
                    "/api/v1/prompts",
                    headers=users["a"]["headers"],
                    json={"name": "shared-name", "content": f"content {index}"},
                )
            finally:
                await ac.aclose()

        results = await asyncio.gather(create(1), create(2))
        for resp in results:
            assert resp.status_code == 201
        assert results[0].json()["id"] != results[1].json()["id"]
        assert results[0].json()["latest_version"] == 1
        assert results[1].json()["latest_version"] == 1
