import io
import cv2
import json
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form
from starlette.responses import StreamingResponse

from app.utils.category_classes import *
from app.utils.anime import animefy
from app.utils.profile import animefy_profile
from app.utils.extract_category import create_feed

router = APIRouter(
    prefix="/v1/api",
)


@router.post("/profile")
def post_profile(file: UploadFile = File(...)):
    image = animefy_profile(file.file)
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@router.post("/animefy")
def post_animefy(file: UploadFile = File(...), text: str = Form(...)):
    image = Image.open(file.file)
    category, caption = create_feed(image, text)
    category = json.dumps(category)
    image = animefy(np.array(image))
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(
        io.BytesIO(image.tobytes()),
        media_type="image/png",
        headers={"x-mtvs-category": category, "x-mtvs-caption": caption},
    )
