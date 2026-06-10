from fastapi import FastAPI

from api.legacy import router as legacy_router
from api.v1.health import health as health_endpoint
from api.v1.router import router as v1_router

app = FastAPI(
    title="IQGS RAG Service",
    version="1.0.0",
    description="Interview RAG API — knowledge ingest, plan, generate questions.",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

app.include_router(v1_router, prefix="/api/v1")
app.include_router(legacy_router)

app.add_api_route(
    "/health",
    health_endpoint,
    methods=["GET"],
    tags=["Health"],
    summary="Health check (root, for probes)",
    include_in_schema=True,
)
