import json

import torch
from transformers import RobertaTokenizer

with open("config/config.json") as f:
    config = json.load(f)

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

AUG = config["img_augmentation"]  # origin img -> AUG(number) imgs

transform = transforms.Compose(
    [
        transforms.Resize(300),
        transforms.RandomResizedCrop(224, scale=(0.5, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MM_Dataset(Dataset):
    def __init__(self, is_train=True) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.is_train = is_train

        print("loading data")

        if is_train == True:
            _data = pd.read_csv("data/train.txt")
        else:
            _data = pd.read_csv("data/test_without_label.txt")

        if config["use_all_data"] == False:
            _data = _data.iloc[0 : config["train_time_data_size"]]

        self.data = {}

        for guid in _data.guid:
            self.data[guid] = {}

            if self.is_train == True:
                tag = _data.loc[_data.guid == guid, "tag"].iloc[0]
                if tag == "positive":
                    self.data[guid]["tag"] = 0
                elif tag == "neutral":
                    self.data[guid]["tag"] = 1
                elif tag == "negative":
                    self.data[guid]["tag"] = 2
            else:
                self.data[guid]["tag"] = None

        del _data

        self.imgs = []
        print("preprocessing data")
        for guid in tqdm(self.data.keys()):
            text_path = "data/data/" + str(guid) + ".txt"

            with open(text_path, errors="ignore") as f:
                self.data[guid]["text"] = self.tokenizer.encode_plus(
                    f.readline().strip(),
                    max_length=config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            self.data[guid]["text"] = dict(
                input_ids=self.data[guid]["text"].input_ids.squeeze(0),
                attention_mask=self.data[guid]["text"].attention_mask.squeeze(0),
            )

            img = Image.open("data/data/" + str(guid) + ".jpg")
            for _ in range(AUG):
                self.imgs.append([guid, transform(img)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, _id):
        guid = self.imgs[_id][0]
        img = self.imgs[_id][1]
        text = self.data[guid]["text"]
        tag = torch.zeros(3)
        if self.is_train == True:
            tag[self.data[guid]["tag"]] = 1
            return dict(guid=guid, img=img, text=text, tag=tag)
        else:
            return dict(guid=guid, img=img, text=text)


def train_data():
    train_data = MM_Dataset()
    train_size = int(0.95 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True)
    return train_loader, val_loader


def test_data():
    test_data = MM_Dataset(is_train=False)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"])
    return test_loader
