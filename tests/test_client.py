# ------------------------------------------------------------------
# 1️⃣  Imports – everything we’ll use in the test‑suite.
# ------------------------------------------------------------------
import asyncio
from pathlib import Path

import pytest
from bson import ObjectId

from sapien import SapienClient, SapienConfig, CollectionNames


# ------------------------------------------------------------------
# 2️⃣  A helper that checks whether a service is reachable.
#
#     We keep the tests light – if MongoDB or Qdrant are not running,
#     we simply skip the test instead of failing hard.  This makes
#     CI‑friendly and keeps the repository self‑contained.
# ------------------------------------------------------------------
def _is_service_up(url: str) -> bool:
    """
    Very small ping helper – tries to open a TCP socket.
    Works for `mongodb://…` and `http://…` URLs.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False


# ------------------------------------------------------------------
# 3️⃣  Fixture – create a fresh SapienClient that talks to a
#      temporary database (`sapien_test`).  The client is closed
#      automatically after the test module finishes.
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
async def client() -> SapienClient:
    # Skip if we cannot reach Mongo or Qdrant – let CI decide whether
    # to run integration tests.
    if not _is_service_up("mongodb://localhost:27017"):
        pytest.skip("MongoDB is not reachable (localhost:27017)")
    if not _is_service_up("http://localhost:6333"):
        pytest.skip("Qdrant is not reachable (localhost:6333)")

    cfg = SapienConfig(
        mongo_uri="mongodb://localhost:27017",
        db_name="sapien_test",
        qdrant_url="http://localhost:6333",
        collections=CollectionNames(),
    )

    async with SapienClient(cfg) as c:
        # Ensure a clean slate
        await c._db.drop_collection(c._col_names.messages)
        yield c


# ------------------------------------------------------------------
# 4️⃣  Test – add_message() stores a document correctly.
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_add_message(client: SapienClient):
    msg_id = await client.add_message("sess1", "user", "Hello world")
    assert isinstance(msg_id, ObjectId)

    doc = await client._messages.find_one({"_id": msg_id})
    assert doc is not None
    assert doc["content"] == "Hello world"
    assert doc["role"] == "user"


# ------------------------------------------------------------------
# 5️⃣  Test – get_context() returns at least one relevant document.
#      We give the async embedding task a short moment to run.
# ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_context(client: SapienClient):
    # Add a couple of messages so that there is something to search for
    await client.add_message("sess1", "assistant", "I am here.")
    await client.add_message("sess1", "user", "What can you do?")

    # Give the background embedding task a tiny window
    await asyncio.sleep(0.5)

    ctx = await client.get_context("sess1", "assist")
    assert isinstance(ctx, list)
    assert len(ctx) >= 1

    # The returned docs should contain the original content field
    for d in ctx:
        assert "content" in d


# ------------------------------------------------------------------
# 6️⃣  Optional sanity – check that collections are correctly namespaced.
# ------------------------------------------------------------------
def test_collection_namespaces():
    col = CollectionNames()
    assert col.sessions == "sapien_sessions"
    assert col.messages == "sapien_messages"
 
