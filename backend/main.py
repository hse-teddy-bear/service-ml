from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from db import init_db
from routers.forward import router as forward_router
from routers.health import router as health_router


app = FastAPI(title="Stock Sentiment Inference Service")


@app.on_event("startup")
def on_startup() -> None:
    # DB tables
    init_db()
    # Prometheus metrics
    Instrumentator().instrument(app).expose(app)


app.include_router(forward_router)
app.include_router(health_router)
