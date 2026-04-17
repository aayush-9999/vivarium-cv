# api/middleware.py
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("vivarium.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI app in one call."""

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing + logging ──────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%s %s → %d  (%d ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response