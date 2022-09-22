import torch
import numpy as np
from PIL import Image

import app.stylegan2 as stylegan2
from app.stylegan2 import utils
from app.stylegan2.utils import ImageFolder
from app.ffhq_dataset.face_alignment import image_align
from app.ffhq_dataset.landmarks_detector import LandmarksDetector

pixel_min = -1.0
pixel_max = 1.0
output_size = 512
num_steps = 200
seed = 1234
landmarks_model_path = "app/models/shape_predictor_68_face_landmarks.dat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
landmarks_detector = LandmarksDetector(landmarks_model_path)
G_PRE = stylegan2.models.load("app/models/G_pretrained.pth")
G_PRE = utils.unwrap_module(G_PRE).to(device)
G = stylegan2.models.load("app/models/G_blend.pth")
G = G.G_synthesis
G = G.to(device)


def animefy_profile(file):
    image = Image.open(file)
    img_name = "temp." + image.format
    image.save("app/images/raw/" + img_name)
    raw_img_path = "app/images/raw/" + img_name
    aligned_img_path = "app/images/aligned/real_face/temp.png"
    for face_landmarks in landmarks_detector.get_landmarks(raw_img_path):
        image_align(raw_img_path, aligned_img_path, face_landmarks, output_size=512)

    dataset = ImageFolder(
        "app/images/aligned", pixel_min=pixel_min, pixel_max=pixel_max
    )
    rnd = np.random.RandomState(seed)
    indices = rnd.choice(len(dataset), size=len(dataset), replace=False)
    images = []
    for i in indices:
        data = dataset[i]
        if isinstance(data, (tuple, list)):
            data = data[0]
        images.append(data)
    images = torch.stack(images).to(device)

    lpips_model = stylegan2.external_models.lpips.LPIPS_VGG16(
        pixel_min=pixel_min, pixel_max=pixel_max
    )

    proj = stylegan2.project.Projector(
        G=G_PRE,
        dlatent_device=device,
        lpips_model=lpips_model,
        dlatent_avg_samples=10000,
        dlatent_batch_size=1024,
        lpips_size=256,
    )

    for i in range(0, len(images), 1):
        target = images[i : i + 1]
        proj.start(
            target=target,
            num_steps=num_steps,
            verbose=True,
        )

        for _ in range(num_steps):
            proj.step()

    latents = proj.get_dlatent().cpu().detach().numpy()
    latent = latents[0]
    np.save(f"app/images/latent/temp.npy", latent)
    latent = np.load("app/images/latent/temp.npy", allow_pickle=True)
    latent = torch.tensor(latent[np.newaxis, ...]).to(device)
    out = G(latent)
    out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]

    return np.array(out)
