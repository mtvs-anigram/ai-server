import io
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form
from starlette.responses import StreamingResponse

from app.utils.anime import animefy

router = APIRouter(
    prefix="/v1/api",
)


@router.post("/profile")
def post_profile(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@router.post("/animefy")
def post_animefy(file: UploadFile = File(...), text: str = Form(...)):
    image = np.array(Image.open(file.file))
    image = animefy(image)
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(
        io.BytesIO(image.tobytes()),
        media_type="image/png",
        headers={"x-mtvs-category": "category", "x-mtvs-caption": "caption"},
    )
