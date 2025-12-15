import base64
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from ..db import log_inference
from ..model import predict_text


class TextRequest(BaseModel):
    text: str


router = APIRouter()


@router.post("/forward")
async def forward(
    text_request: Optional[TextRequest] = None,
    image: Optional[UploadFile] = File(default=None),
    x_extra_param: Optional[str] = Header(default=None, alias="X-Extra-Param"),
):
    """
    Unified /forward endpoint.

    - If `image` is not provided → expect JSON with `text`.
    - If `image` is provided → expect multipart/form-data with image file.
      Any additional params should come from headers (example: X-Extra-Param).
    """

    # Wrong format: neither JSON text nor image provided
    if text_request is None and image is None:
        raise HTTPException(status_code=400, detail="bad request")

    # Branch without image: plain text sentiment
    if image is None and text_request is not None:
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

    # Branch with image:
    # For демонстрация, просто возвращаем base64 того же изображения + метаданные из хэдера.
    if image is not None:
        try:
            raw_bytes = await image.read()
            img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            response = {
                "image_base64": b64_img,
                "filename": image.filename,
                "content_type": image.content_type,
                "extra_param": x_extra_param,
            }
            log_inference(
                text=None,
                label=None,
                probs=None,
                has_image=True,
                status="ok",
            )
            return JSONResponse(content=response)
        except Exception:
            log_inference(
                text=None,
                label=None,
                probs=None,
                has_image=True,
                status="failed",
            )
            raise HTTPException(
                status_code=403, detail="модель не смогла обработать данные"
            )


