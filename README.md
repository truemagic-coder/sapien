# Sapien – Long Term Memory for AI

Sapien is a lightweight Python client that lets you:

* Persist chat messages (or any event) in **MongoDB**.
* Compute dense embeddings with **sentence‑transformers**.
* Store those vectors in **Qdrant** for fast semantic search.
* Retrieve the most relevant historical context for a given query.

It’s ideal for building LLM‑powered assistants that need to remember past conversations or knowledge graphs while keeping all data stored in open‑source databases.

> ⚠️  The library is still a work‑in‑progress.  It has minimal tests and expects MongoDB + Qdrant to be running locally (or reachable from your environment).

---

## Features

| Feature | Status |
|---------|--------|
| **MongoDB persistence** | ✅ |
| **Qdrant vector search** | ✅ |
| **Sentence‑transformers embeddings** | ✅ |
| **Convenient async API** | ✅ |
| **Zero‑configuration defaults (except for services)** | ✅ |
| **Type‑hinted, testable code** | ✅ |

---

## Quick start

```bash
# 1️⃣ Install the package + dev deps
poetry install

# 2️⃣ Start MongoDB and Qdrant locally
#    (use Docker Compose – see docker-compose.yml)
docker compose up -d

# 3️⃣ Run a short demo script
python demo.py
```

**demo.py**

```python
import asyncio
from datetime import datetime

from sapien import SapienClient, SapienConfig, CollectionNames


async def main():
    cfg = SapienConfig(
        mongo_uri="mongodb://localhost:27017",
        db_name="sapien",
        qdrant_url="http://localhost:6333",
        collections=CollectionNames(),
    )
    async with SapienClient(cfg) as db:
        await db.init_indexes()

        # Add a new message
        msg_id = await db.add_message(
            session_id="chat_42",
            role="user",
            content="I need a laptop for gaming.",
            timestamp=datetime.utcnow()
        )

        # Ask the context for a keyword
        ctx = await db.get_context("chat_42", "laptop")
        print(f"Context ({len(ctx)} docs):")
        for doc in ctx:
            print("-", doc["content"])


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Installation

> **Prerequisites** –  
> *Python 3.12+*  
> *MongoDB server* (>=4.0)  
> *Qdrant server* (>=1.0)

```bash
# Install via Poetry
poetry add sapien
```

If you want the full stack (including the optional `sentence-transformers` and Qdrant client), just install the package normally – all dependencies are pulled in automatically.

---

## Configuration

All configuration is done through a single dataclass:

```python
from sapien import SapienConfig, CollectionNames

cfg = SapienConfig(
    mongo_uri="mongodb://localhost:27017",
    db_name="sapien",          # database name
    qdrant_url="http://localhost:6333",
    collections=CollectionNames(),   # optional custom names
)
```

> **Tip** – The default collection names are prefixed with `sapien_` (`sessions`, `messages`, etc.) to keep them isolated in the `sapien` database.

---

## API Reference

| Method | Description |
|--------|-------------|
| `SapienClient.__aenter__ / __aexit__` | Async context manager that ensures collection creation. |
| `add_message(session_id, role, content, timestamp=None)` | Persist a message and fire‑and‑forget its embedding & Qdrant upsert. Returns the Mongo `_id`. |
| `get_context(session_id, query, k=10)` | Vector search in Qdrant → return full Mongo docs for the top *k* matches. |
| `init_indexes()` | Create idempotent indexes (`sessions.session_id`, `messages.timestamp`, etc.). |

---

## Testing

The project ships with a small async test‑suite that expects MongoDB and Qdrant to be running locally.

```bash
# Run tests
poetry run pytest -vv
```

If the services are not reachable, the integration tests will be skipped automatically.

---

## Development

1. **Clone & install**

   ```bash
   git clone https://github.com/yourname/sapien.git
   cd sapien
   poetry install
   ```

2. **Run linters / formatters**

   ```bash
   poetry run ruff check .
   poetry run black src tests
   ```

3. **Run the demo**

   ```bash
   python demo.py
   ```

4. **Add a new feature** – remember to update `pyproject.toml`, write tests, and add documentation.

---

## Contributing

Pull requests are welcome!  
Please:

1. Fork the repo.
2. Create a feature branch (`feature/your-feature`).
3. Write or update tests.
4. Run `poetry run pytest`.
5. Submit a PR.

For major changes, open an issue first to discuss the scope.

---

## License

MIT © 2025 – feel free to use it however you like.

---

