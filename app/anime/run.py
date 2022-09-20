from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.anime.util import singleton


def animefy(img: np.ndarray, style: str = "Shinkai") -> np.ndarray:
    model = ModelManager().models[style]
    img, scale = load(img)
    res = convert(img, scale, model)
    return res


@singleton
class ModelManager:
    styles = ("Hayao", "Shinkai", "Paprika")

    def __init__(self) -> None:
        self._models = None

    def load(self) -> None:
        device_name = ort.get_device().upper()
        if device_name == "CPU":
            providers = ["CPUExecutionProvider"]
        elif device_name == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsupported device")
        # providers = ["CPUExecutionProvider"]
        # pylint: disable=unnecessary-lambda-assignment
        path = lambda s: Path(__file__).parent / "models" / f"AnimeGANv2_{s}.onnx"
        self._models = {
            s: ort.InferenceSession(str(path(s)), providers=providers)
            for s in self.styles
        }

    @property
    def models(self) -> dict[str, ort.InferenceSession]:
        if self._models is None:
            self.load()
        return self._models


def fit(img: np.ndarray, x32: bool = True) -> np.ndarray:
    def to_32s(x):
        return 256 if x < 256 else x - x % 32

    h, w = img.shape[:2]
    if x32:  # resize image to multiple of 32s
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    img = img.astype(np.float32) / 127.5 - 1.0
    return img


def load(img0: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    img = fit(img0)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]


def convert(
    img: np.ndarray, scale: tuple[int, int], session: ort.InferenceSession
) -> np.ndarray:
    x = session.get_inputs()[0].name
    # y = session.get_outputs()[0].name
    fake_img = session.run(None, {x: img})[0]
    images = (np.squeeze(fake_img) + 1.0) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, (scale[1], scale[0]))
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
