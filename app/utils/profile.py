import torch

import app.stylegan2 as stylegan2

pixel_min = 1.0
pixel_max = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = stylegan2.models.load("app/models/G_blend.pth")
G = G.to(device)


def project_images(image):
    image = torch.from_numpy(image).to(device)

    lpips_model = stylegan2.external_models.lpips.LPIPS_VGG16(
        pixel_min=pixel_min, pixel_max=pixel_max
    )

    proj = stylegan2.project.Projector(
        G=G,
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

    return image.cpu().numpy()
