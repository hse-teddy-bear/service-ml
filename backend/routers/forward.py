from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db import log_inference
from model import predict_text


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

    try:
        result = predict_text(text_request.text)
        log_inference(
            text=text_request.text,
            label=result["label"],
            probs=result["probs"],
            has_image=False,
            status="ok",
        )
    except RuntimeError:
        log_inference(
            text=text_request.text,
            label=None,
            probs=None,
            has_image=False,
            status="failed",
        )
        raise HTTPException(
            status_code=403, detail="модель не смогла обработать данные"
        )
    return JSONResponse(content=result)


