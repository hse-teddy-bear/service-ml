from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from db import init_db
from routers.forward import router as forward_router
from routers.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: инициализация БД
    init_db()
    yield
    # Shutdown: здесь можно добавить код для завершения работы


app = FastAPI(
    title="Stock Sentiment Inference Service",
    lifespan=lifespan
)

# Prometheus middleware должен быть добавлен ДО подключения роутеров
Instrumentator().instrument(app).expose(app)


app.include_router(forward_router)
app.include_router(health_router)
