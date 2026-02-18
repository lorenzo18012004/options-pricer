"""
Contexte de requête avec correlation ID pour le tracing.
"""

import logging
import uuid
from contextvars import ContextVar

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Retourne l'ID de requête courant."""
    return _request_id_var.get() or ""


def set_request_id(rid: str = None) -> str:
    """Définit l'ID de requête. Retourne l'ID."""
    rid = rid or str(uuid.uuid4())[:8]
    _request_id_var.set(rid)
    return rid


def log_with_request_id(logger: logging.Logger, level: int, msg: str, *args, **kwargs):
    """Log avec request_id dans extra."""
    rid = get_request_id()
    extra = kwargs.pop("extra", {})
    extra["request_id"] = rid
    kwargs["extra"] = extra
    logger.log(level, msg, *args, **kwargs)
