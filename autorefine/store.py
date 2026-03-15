"""Store facade — re-exports for convenience."""

from autorefine.storage.base import BaseStore
from autorefine.storage.json_store import JSONStore

__all__ = ["BaseStore", "JSONStore"]
