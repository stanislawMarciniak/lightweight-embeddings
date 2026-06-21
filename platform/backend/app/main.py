from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import chat, conversations, faq, documents
from app.services.semantic_model import load_encoder, reload_encoder, resolve_model_path
from app.utils.wordpiece import encode_texts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the semantic encoder once at startup and keep it in memory."""
    settings = get_settings()
    if settings.EMBEDDING_BACKEND == "custom":
        t0 = time.perf_counter()
        try:
            load_encoder(resolve_model_path(settings))
            logger.info("Model loading time: %.1f ms", (time.perf_counter() - t0) * 1000.0)
            t1 = time.perf_counter()
            encode_texts(["warmup"])
            logger.info("Tokenizer warmup time: %.1f ms", (time.perf_counter() - t1) * 1000.0)
        except Exception:  # noqa: BLE001 - log but don't block startup
            logger.exception("Failed to load semantic encoder")
    yield

_DEV_VITE_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]


def _cors_origins() -> List[str]:
    """Allowed browser origins (must match CORSMiddleware and error handlers).

    ``FRONTEND_ORIGIN`` may be a single origin or comma-separated list. Local Vite
    dev ports (5173/5174) are always merged in so a stale ``5173`` env value does
    not block the app when Vite falls back to ``5174``.
    """
    settings = get_settings()
    configured: List[str] = []
    if settings.FRONTEND_ORIGIN:
        configured = [o.strip() for o in settings.FRONTEND_ORIGIN.split(",") if o.strip()]
    seen: set[str] = set()
    merged: List[str] = []
    for origin in configured + _DEV_VITE_ORIGINS:
        if origin not in seen:
            seen.add(origin)
            merged.append(origin)
    return merged


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Semantic Chat Application", version="1.0.0", lifespan=lifespan)

    # CORS
    origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _cors_headers(request: Request) -> dict:
        origin = request.headers.get("origin")
        if origin in origins:
            allow_origin = origin
        else:
            allow_origin = origins[0] if origins else "*"
        return {
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Credentials": "true",
        }

    @app.exception_handler(HTTPException)
    async def http_exception_with_cors(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=_cors_headers(request),
        )

    @app.exception_handler(Exception)
    async def add_cors_to_errors(request: Request, exc: Exception):
        """Ensure error responses include CORS headers so the browser does not block them."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
            headers=_cors_headers(request),
        )

    # Routers
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(faq.router)
    app.include_router(documents.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/admin/reload-model")
    async def reload_model() -> dict:
        """Hot-reload encoder weights from disk (call after `manage.py finetune`)."""
        import time as _t

        t0 = _t.perf_counter()
        reload_encoder(resolve_model_path(settings))
        return {"status": "reloaded", "reload_ms": round((_t.perf_counter() - t0) * 1000.0, 1)}

    @app.get("/metrics")
    async def metrics() -> dict:
        from app.services.semantic_cache import get_semantic_cache
        from app.services.semantic_model import get_encoder

        enc = get_encoder()
        return {
            "embedding_backend": settings.EMBEDDING_BACKEND,
            "embedding_dim": (enc.output_dim if enc else 128),
            "semantic_cache": get_semantic_cache().stats(),
        }

    return app


app = create_app()

