import pickle
import torch
from urllib import parse
from googletrans import Translator
from konlpy.tag import Okt
from torchvision import transforms

from app.utils.category_classes import *
from app.utils.util import list_to_string, load_image

okt = Okt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_path = "app/models/encoder-5.ckpt"
decoder_path = "app/models/decoder-5.ckpt"
vocab_path = "app/models/vocab.pkl"

embed_size = 256
hidden_size = 512
num_layers = 1

encoder = EncoderCNN(embed_size).eval()
encoder = encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path))


def create_feed(img, main_txt):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        f.close()

    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    decoder = decoder.to(device)
    decoder.load_state_dict(torch.load(decoder_path))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    image = load_image(img, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    sampled_caption = []
    for word_id in sampled_ids:  # 하나씩 단어 인덱스를 확인하며
        word = vocab.idx2word[word_id]  # 단어 문자열로 바꾸어 삽입
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption)

    translator = Translator()
    caption_final = translator.translate(sentence, dest="ko")
    tmp = caption_final.text
    tmp = tmp.split(" ")

    result = list_to_string(tmp[1:6])
    total_txt = result + " " + main_txt
    total_txt = okt.nouns(total_txt)
    for i, v in enumerate(total_txt):
        if len(v) == 1:
            total_txt.pop(i)

    total_txt = [parse.quote(i) for i in total_txt]

    return total_txt, parse.quote(result)
