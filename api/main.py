# api/main.py
from fastapi import FastAPI
from api.routes import infer, pipeline
from api.middleware import register_middleware

app = FastAPI(
    title="Vivarium CV API",
    version="0.2.0",
    description=(
        "Computer vision system for vivarium cage monitoring. "
        "Mouse detection, water and food level tracking.\n\n"
        "## Workflows\n"
        "All data-pipeline operations (labelling, augmentation, training, validation) "
        "are available via `/pipeline/*` endpoints in addition to the Python orchestrator."
    ),
)

register_middleware(app)

# Core inference (existing)
app.include_router(infer.router)

# Full pipeline (new — wraps all former scripts)
app.include_router(pipeline.router)