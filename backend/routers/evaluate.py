from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from services.inference import run_evaluate


router = APIRouter()


@router.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    """
    Принимает .csv файл с колонками `text` и `target` (0,1,2),
    прогоняет модель и возвращает предсказания + метрики.
    """
    return JSONResponse(content=run_evaluate(file))


