import io
import cv2
import json
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form
from starlette.responses import StreamingResponse

from app.utils.anime import animefy
from app.utils.profile import project_images

router = APIRouter(
    prefix="/v1/api",
)


@router.post("/profile")
def post_profile(file: UploadFile = File(...)):
    image = project_images(file.file)
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/png")


@router.post("/animefy")
def post_animefy(file: UploadFile = File(...), text: str = Form(...)):
    image = np.array(Image.open(file.file))
    # print(text)
    # category, caption = jwh_model(image, text)
    category = ["category"]
    caption = "caption"
    category = json.dumps(category)
    image = animefy(image)
    _, image = cv2.imencode(".png", image)

    return StreamingResponse(
        io.BytesIO(image.tobytes()),
        media_type="image/png",
        headers={"x-mtvs-category": category, "x-mtvs-caption": caption},
    )
