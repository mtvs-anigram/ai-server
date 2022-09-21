import torch
import numpy as np
from PIL import Image

import app.stylegan2 as stylegan2
from app.stylegan2 import utils
from app.ffhq_dataset.face_alignment import image_align
from app.ffhq_dataset.landmarks_detector import LandmarksDetector

pixel_min = 1.0
pixel_max = 1.0
output_size = 512
landmarks_model_path = "app/models/shape_predictor_68_face_landmarks.dat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
landmarks_detector = LandmarksDetector(landmarks_model_path)
G_PRE = stylegan2.models.load("app/models/G_pretrained.pth").to(device)
G = stylegan2.models.load("app/models/G_blend.pth")
G = G.G_synthesis
G = G.to(device)


def project_images(file):
    image = Image.open(file)
    img_name = "temp." + image.format
    image.save("app/images/raw/" + img_name)
    raw_img_path = "app/images/raw/" + img_name
    aligned_img_path = "app/images/aligned/temp.png"

    for face_landmarks in landmarks_detector.get_landmarks(raw_img_path):
        image_align(raw_img_path, aligned_img_path, face_landmarks, output_size)

    image = Image.open(aligned_img_path)
    image = np.array(image)
    image = torch.from_numpy(image).to(device)

    lpips_model = stylegan2.external_models.lpips.LPIPS_VGG16(
        pixel_min=pixel_min, pixel_max=pixel_max
    )

    proj = stylegan2.project.Projector(
        G=G_PRE,
        dlatent_device=device,
        lpips_model=lpips_model,
        dlatent_avg_samples=2500,
        dlatent_batch_size=256,
        lpips_size=64,
    )

    proj.start(
        target=image,
        verbose=True,
    )

    latents = proj.get_dlatent().cpu().detach().numpy()
    np.save("app/images/latent/temp.npy", latents[0])
    latent = np.load("app/images/latent/temp.npy", allow_pickle=True)
    latent = torch.tensor(latent[np.newaxis, ...]).to(device)
    out = G(latent)
    out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]

    return np.array(out)
