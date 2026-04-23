# api/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import infer, cages
from api.middleware import register_middleware
from db.session import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on server startup before any requests are handled.
    Creates DB tables if they don't exist yet.
    In production, replace create_tables() with proper Alembic migrations.
    """
    await create_tables()
    yield


app = FastAPI(
    title="Vivarium CV API",
    version="0.3.0",
    lifespan=lifespan,
    description=(
        "Computer vision system for vivarium cage monitoring.\n\n"
        "## What this API does\n"
        "- Accepts camera frames and runs mouse detection + water/food level estimation\n"
        "- Stores every reading to the database\n"
        "- Exposes cage status, history, and alerts for dashboards and alert systems\n\n"
        "## What this API does NOT do\n"
        "Model training, dataset augmentation, and labelling are **not** exposed here.\n"
        "Run those via the orchestrator or CLI scripts in `scripts/`:\n"
        "```\n"
        "from pipeline.pipeline_factory import get_orchestrator\n"
        "orch = get_orchestrator()\n"
        "orch.run_full_pipeline()\n"
        "```\n"
    ),
)

register_middleware(app)

# Core inference — camera sends a frame, gets DetectionResult back
app.include_router(infer.router)

# Cage monitoring — query cage state, history, and alerts
app.include_router(cages.router)