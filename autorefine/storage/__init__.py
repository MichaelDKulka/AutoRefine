"""Storage backends for AutoRefine.

Use :func:`get_store` to instantiate a backend from configuration::

    from autorefine.storage import get_store

    store = get_store("json")                              # zero-config default
    store = get_store("sqlite", path="/tmp/my.db")         # explicit path
    store = get_store("postgres", database_url="postgresql://...")

Or pass an :class:`~autorefine.config.AutoRefineSettings` object::

    from autorefine.config import AutoRefineSettings
    store = get_store(config=AutoRefineSettings())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autorefine.storage.base import BaseStore
from autorefine.storage.json_store import JSONStore

if TYPE_CHECKING:
    from autorefine.config import AutoRefineSettings


def get_store(
    backend: str = "",
    *,
    path: str = "",
    database_url: str = "",
    config: AutoRefineSettings | None = None,
    **kwargs: Any,
) -> BaseStore:
    """Factory that returns a configured storage backend.

    Args:
        backend: ``"json"``, ``"sqlite"``, or ``"postgres"``.
            When empty, resolved from *config* (or defaults to ``"json"``).
        path: File path for json/sqlite backends.
        database_url: PostgreSQL connection string (postgres only).
        config: An :class:`~autorefine.config.AutoRefineSettings` object.
            If provided, *backend*, *path*, and *database_url* are read
            from the config unless explicitly overridden.
        **kwargs: Extra arguments forwarded to the backend constructor.

    Returns:
        A ready-to-use :class:`BaseStore` instance.

    Raises:
        ValueError: If the backend name is unrecognised.
        ImportError: If a required optional dependency is missing.
    """
    # Resolve from config when not explicitly provided
    namespace = ""
    encryption_key = ""
    if config is not None:
        backend = backend or config.storage_backend
        path = path or config.get_store_path()
        database_url = database_url or config.database_url
        namespace = config.namespace
        encryption_key = config.encryption_key

    backend = backend or "json"

    if backend == "json":
        return JSONStore(path or None, namespace=namespace,
                         encryption_key=encryption_key, **kwargs)

    if backend == "sqlite":
        from autorefine.storage.sqlite_store import SQLiteStore

        store_path = path
        if store_path and store_path.endswith(".json"):
            store_path = store_path.replace(".json", ".db")
        return SQLiteStore(store_path or None, **kwargs)

    if backend == "postgres":
        from autorefine.storage.postgres_store import PostgresStore

        if not database_url:
            raise ValueError(
                "database_url is required for the postgres backend. "
                "Set AUTOREFINE_DATABASE_URL or pass database_url=."
            )
        return PostgresStore(database_url, **kwargs)

    raise ValueError(
        f"Unknown storage backend '{backend}'. "
        f"Supported: json, sqlite, postgres"
    )


__all__ = ["BaseStore", "JSONStore", "get_store"]
