from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.inference import run_forward


class TextRequest(BaseModel):
    text: str


router = APIRouter()


@router.post("/forward")
async def forward(
    text_request: Optional[TextRequest] = None,
):
    """
    /forward endpoint that принимает JSON с полем `text`.
    Пример тела запроса:

    {
        "text": "какой-то текст"
    }
    """

    # Wrong format: JSON не передан вообще
    if text_request is None:
        raise HTTPException(status_code=400, detail="bad request")

    # Проверка на пустой текст
    if not text_request.text:
        raise HTTPException(status_code=400, detail="bad request")

    result = run_forward(text_request.text)
    return JSONResponse(content=result)

