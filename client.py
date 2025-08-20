# src/sapien/client.py
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import bson
import motor.motor_asyncio
import numpy as np
from pymongo.collection import Collection

# ------------------------------------------------------------------
# 1️⃣  The embedding model is a hard dependency – we raise immediately
#     if the package cannot be imported.
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    raise RuntimeError(
        "The 'sentence-transformers' package must be installed to use SapienClient."
    )

# ------------------------------------------------------------------
# 2️⃣  Qdrant async imports – we also require a valid URL.
from qdrant_client import AsyncQdrantClient, models

# ------------------------------------------------------------------
@dataclass(frozen=True)
class CollectionNames:
    """
    All collection names are prefixed with ``sapien_`` so that they
    live in the dedicated ``sapien`` database.  No defaults – you must
    pass an instance explicitly.
    """
    sessions: str = "sapien_sessions"
    messages: str = "sapien_messages"
    entities: str = "sapien_entities"
    relations: str = "sapien_relations"


@dataclass(frozen=True)
class SapienConfig:
    """
    All parameters are required; there are no implicit defaults.
    """
    mongo_uri: str
    db_name: Optional[str] = "sapien"  # e.g. "sapien"
    qdrant_url: str  # must be a valid Qdrant endpoint
    collections: Optional[CollectionNames] = CollectionNames()


class SapienClient:
    """
    Async client for the temporal knowledge‑graph.

    Example
    -------
    >>> from sapien import SapienClient, SapienConfig
    >>> cfg = SapienConfig(
    ...     mongo_uri="mongodb://localhost:27017",
    ...     db_name="sapien",
    ...     qdrant_url="http://localhost:6333",
    ...     collections=CollectionNames()
    ... )
    >>> async with SapienClient(cfg) as db:
    ...     await db.add_message("chat_1", "user", "I need a laptop.")
    ...     ctx = await db.get_context("chat_1", "laptop")
    """

    def __init__(self, config: SapienConfig):
        self._cfg = config
        # ---- Mongo ----------------------------------------------------
        self._mongo_client = motor.motor_asyncio.AsyncIOMotorClient(config.mongo_uri)
        self._db: motor.motor_asyncio.AsyncIOMotorDatabase = self._mongo_client[config.db_name]
        self._col_names = config.collections

        # ---- Qdrant -----------------------------------------------
        if not config.qdrant_url:
            raise ValueError("`qdrant_url` must be a non‑empty string")
        self._qdrant: AsyncQdrantClient = AsyncQdrantClient(url=config.qdrant_url)

        # ---- Embedding model -----------------------------------------
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )  # this call will raise if the package is missing

    async def __aenter__(self):
        await self.ensure_collection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the underlying Mongo client."""
        self._mongo_client.close()
        # AsyncQdrantClient has no explicit close() – it just uses aiohttp session.

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist yet."""
        try:
            await self._qdrant.create_collection(
                collection_name="llm_memory",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        except Exception:  # pragma: no cover
            # If the collection already exists we can safely ignore the error
            pass

    async def _embed_and_upsert(self, message_id: bson.ObjectId):
        """Compute vector → store in Mongo (binary) & upsert to Qdrant."""
        msg_doc = await self._messages.find_one({"_id": message_id})
        if not msg_doc:
            return

        loop = asyncio.get_running_loop()
        vec_np: np.ndarray = await loop.run_in_executor(
            None, lambda: self._model.encode(msg_doc["content"])
        )

        # Store as binary in Mongo
        vec_bytes = vec_np.astype(np.float32).tobytes()
        await self._messages.update_one(
            {"_id": message_id},
            {"$set": {"embedding": vec_bytes}},
        )

        # Upsert into Qdrant (async)
        point = models.PointStruct(
            id=str(message_id),
            vector=vec_np.tolist(),
            payload={"session_id": msg_doc["session_id"]},
        )
        await self._qdrant.upsert(collection_name="llm_memory", points=[point])

    # ------------------------------------------------------------------
    # Public API -------------------------------------------------------
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> bson.ObjectId:
        """
        Persist a raw message and asynchronously compute its vector.
        Returns the Mongo `_id` of the inserted document.
        """
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        ts = timestamp or datetime.utcnow()

        msg_doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": ts,
        }

        result = await self._messages.insert_one(msg_doc)
        message_id = result.inserted_id

        # Fire‑and‑forget: embedding + Qdrant upsert
        asyncio.create_task(self._embed_and_upsert(message_id))

        return message_id

    async def get_context(
        self,
        session_id: str,
        query: str,
        k: int = 10,
    ) -> List[dict]:
        """
        Vector search in Qdrant → fetch the full Mongo docs.
        Requires that a Qdrant URL was supplied at construction.
        """
        # Encode query in background
        loop = asyncio.get_running_loop()
        vec_np: np.ndarray = await loop.run_in_executor(
            None,
            lambda: self._model.encode(query),
        )
        q_vec = vec_np.tolist()

        hits = await self._qdrant.search(
            collection_name="llm_memory",
            query_vector=q_vec,
            limit=k,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id", match=models.MatchValue(value=session_id)
                    )
                ]
            ),
        )

        ids = [hit.id for hit in hits]
        cursor = self._messages.find({"_id": {"$in": [bson.ObjectId(i) for i in ids]}})
        return await cursor.to_list(length=None)

    # ------------------------------------------------------------------
    # Convenience properties ---------------------------------------------
    @property
    def _sessions(self) -> Collection:
        return self._db[self._col_names.sessions]

    @property
    def _messages(self) -> Collection:
        return self._db[self._col_names.messages]

    @property
    def _entities(self) -> Collection:
        return self._db[self._col_names.entities]

    @property
    def _relations(self) -> Collection:
        return self._db[self._col_names.relations]

    # ------------------------------------------------------------------
    # Index creation (call once after construction)
    async def init_indexes(self):
        """Create the default indexes (idempotent)."""
        await self._sessions.create_index("session_id", unique=True)

        await self._messages.create_index(
            [("session_id", 1), ("timestamp", -1)]
        )
        # Keep history for 30 days
        await self._messages.create_index(
            [("timestamp", -1)], expireAfterSeconds=60 * 60 * 24 * 30
        )

    # ------------------------------------------------------------------
    # Optional: expose raw client objects (if you need them)
    @property
    def mongo_client(self):
        return self._mongo_client

    @property
    def qdrant_client(self):
        return self._qdrant
