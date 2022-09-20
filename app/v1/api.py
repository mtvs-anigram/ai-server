import io

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

from app.anime import animefy

router = APIRouter(
    prefix="/v1/api",
)


@router.get("/")
def index():
    return {"message": "Hello World"}


@router.post("/profile")
def post_profile(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    _, image = cv2.imencode(".png", image)  # pylint: disable=no-member
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@router.post("/category")
def post_category():
    return {"category": "cat", "caption": "A cat"}


@router.post("/animefy")
def post_animefy(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = animefy(image)
    _, image = cv2.imencode(".png", image)  # pylint: disable=no-member
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")
