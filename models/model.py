import torch
from torch import nn
from transformers import RobertaModel


class Text_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        return output


class Img_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        )
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.resnet(x)


class MM_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = Text_Model()
        self.img_model = Img_Model()
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(6, 3)

    def forward(self, item):
        has_text = "text" in item.keys()
        has_img = "img" in item.keys()
        text = item["text"] if has_text else None
        img = item["img"] if has_img else None

        text_probs = 0
        img_probs = 0

        text_probs = (
            self.text_model(
                input_ids=text["input_ids"], attention_mask=text["attention_mask"]
            )
            if has_text
            else torch.zeros((item["img"].shape[0], 3)).to(item["img"].device)
        )

        img_probs = (
            self.img_model(img)
            if has_img
            else torch.zeros((item["text"]["input_ids"].shape[0], 3)).to(
                item["text"]["input_ids"].device
            )
        )

        output = text_probs + img_probs

        concat = torch.cat((text_probs, img_probs), 1)
        concat = self.relu(concat)
        output += self.layer_1(concat)

        return output
