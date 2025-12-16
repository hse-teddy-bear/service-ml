from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from db import init_db
from routers.evaluate import router as evaluate_router
from routers.forward import router as forward_router
from routers.forward_batch import router as forward_batch_router
from routers.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Stock Sentiment Inference Service",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)


app.include_router(forward_router)
app.include_router(forward_batch_router)
app.include_router(evaluate_router)
app.include_router(health_router)
