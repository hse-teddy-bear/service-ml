from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from services.inference import run_forward_batch


router = APIRouter()


@router.post("/forward_batch")
async def forward_batch(file: UploadFile = File(...)):
    """
    Принимает .csv файл с колонкой `text` и возвращает список предсказаний.
    """
    return JSONResponse(content=run_forward_batch(file))


