from PIL import Image


def load_image(img, transform=None):
    image = img.convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def list_to_string(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()
