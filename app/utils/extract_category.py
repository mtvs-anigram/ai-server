import re
import pickle
import googletrans
from PIL import Image
from konlpy.tag import Mecab

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from app.utils.wrapper import singleton

mec = Mecab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_path = "app/models/encoder-5.ckpt"
decoder_path = "app/models/decoder-5.ckpt"
vocab_path = "app/models/vocab.pkl"
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)


@singleton
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features


@singleton
class DecoderRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20
    ):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def sample(self, features, states=None):
        sampled_indexes = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_indexes.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_indexes = torch.stack(sampled_indexes, 1)

        return sampled_indexes


@singleton
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def load_image(img, transform=None):
    image = img.convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


# embed_size = 0
# hidden_size = 0
# num_layers = 0
encoder = EncoderCNN(embed_size).eval()
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder = encoder.to(device)
decoder = decoder.to(device)

# Load the trained model parameters
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))


def create_feed(img, main_txt):
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
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption)

    translator = googletrans.Translator()
    caption = translator.translate(sentence, dest="ko")
    caption = re.compile("[가-힣]+").findall(caption.text)
    temp = " ".join(caption)

    image = Image.open(img)
    total_txt = temp + " " + main_txt
    category_list = mec.nouns(total_txt)

    return category_list, caption


# path = "/content/drive/MyDrive/data/test3.jpg"
# txt = "오늘 친구와 해변에 가서 너무 재밌었다. 다음에 또 가고 싶다."
# create_feed(path, txt)
# print(create_feed(path, txt))
