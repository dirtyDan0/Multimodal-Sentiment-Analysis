import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def train(
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    epochs=3,
    verbose=True,
):
    global device
    loss_acc = pd.DataFrame(
        columns=[
            "Train Loss",
            "Train Acc",
            "Valid Loss",
            "Valid Acc",
            "Valid Loss(no text)",
            "Valid Acc(no text)",
            "Valid Loss(no image)",
            "Valid Acc(no image)",
        ]
    )

    for i in range(epochs):
        tmp_val_acc = []
        print("Epoch", i)

        model.train()
        Y_pred, Y_real, losses = [], [], []
        func = tqdm if verbose else lambda x: x
        for item in func(train_loader):
            img = item["img"].to(device)
            text = item["text"]
            text["input_ids"] = text["input_ids"].to(device)
            text["attention_mask"] = text["attention_mask"].to(device)
            Y = item["tag"].to(device)
            optimizer.zero_grad()
            pred = model({"text": text, "img": img})
            loss = loss_fn(pred, Y)
            losses.append(loss.item())
            Y_pred.extend(pred.argmax(dim=1).to("cpu"))
            Y_real.extend(np.where(Y.to("cpu") == 1)[1])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        Y_pred = np.array(Y_pred)
        Y_real = np.array(Y_real)
        loss_mean = np.array(losses).mean()
        acc = (Y_real == Y_pred).sum() / Y_pred.shape[0]
        train_loss.append(loss_mean)
        train_acc.append(acc)

        print("Train loss :{:.6f}".format(loss_mean), end="\t")
        print("Train Acc  : {:.6f}".format(acc))
        tmp_val_acc.extend([loss_mean, acc])
        tmp_val_acc.extend(
            val(model, loss_fn, val_loader, val_loss, val_acc, verbose=verbose)
        )
        loss_acc.loc[i] = tmp_val_acc
    loss_acc.to_csv("output/loss_and_acc.csv", index_label="Epoch")
    return loss_acc


def val(model, loss_fn, val_loader, val_loss, val_acc, verbose=True):
    global device
    model.eval()
    (
        Y_pred,
        Y_real,
        Y_pred_no_text,
        Y_pred_no_img,
        losses,
        losses_no_text,
        losses_no_img,
    ) = ([], [], [], [], [], [], [])
    with torch.no_grad():
        for item in val_loader:
            img = item["img"].to(device)
            text = item["text"]
            text["input_ids"] = text["input_ids"].to(device)
            text["attention_mask"] = text["attention_mask"].to(device)
            Y = item["tag"].to(device)
            pred = model({"text": text, "img": img})
            pred_no_text = model({"img": img})
            pred_no_img = model({"text": text})
            loss = loss_fn(pred, Y)
            loss_no_text = loss_fn(pred_no_text, Y)
            loss_no_img = loss_fn(pred_no_img, Y)
            losses.append(loss.item())
            losses_no_text.append(loss_no_text.item())
            losses_no_img.append(loss_no_img.item())
            Y_real.extend(np.where(Y.to("cpu") == 1)[1])
            Y_pred.extend(pred.argmax(dim=1).to("cpu"))
            Y_pred_no_text.extend(pred_no_text.argmax(dim=1).to("cpu"))
            Y_pred_no_img.extend(pred_no_img.argmax(dim=1).to("cpu"))

    Y_pred = np.array(Y_pred)
    Y_pred_no_text = np.array(Y_pred_no_text)
    Y_pred_no_img = np.array(Y_pred_no_img)
    Y_real = np.array(Y_real)

    loss_mean = np.array(losses).mean()
    loss_mean_no_text = np.array(losses_no_text).mean()
    loss_mean_no_img = np.array(losses_no_img).mean()
    acc = (Y_real == Y_pred).sum() / Y_pred.shape[0]
    acc_no_text = (Y_real == Y_pred_no_text).sum() / Y_pred.shape[0]
    acc_no_img = (Y_real == Y_pred_no_img).sum() / Y_pred.shape[0]
    val_loss.append(loss_mean)
    val_acc.append(acc)

    if verbose:
        print("Valid Loss : {:.6f}".format(loss_mean), end="\t")
        print("Valid Acc  : {:.6f}".format(acc))
        print("Valid Loss(no text) : {:.6f}".format(loss_mean_no_text), end="\t")
        print("Valid Acc(no text)  : {:.6f}".format(acc_no_text))
        print("Valid Loss(no img) : {:.6f}".format(loss_mean_no_img), end="\t")
        print("Valid Acc(no img)  : {:.6f}".format(acc_no_img))
        print("-" * 45)
    return [
        loss_mean,
        acc,
        loss_mean_no_text,
        acc_no_text,
        loss_mean_no_img,
        acc_no_img,
    ]


def test(model, test_loader):
    global device
    model.eval()
    _data = pd.read_csv("data/test_without_label.txt")
    Y_pred = {}
    with torch.no_grad():
        for item in tqdm(test_loader):
            guid = item["guid"]
            guid = [int(_guid) for _guid in guid]
            img = item["img"].to(device)
            text = item["text"]
            text["input_ids"] = text["input_ids"].to(device)
            text["attention_mask"] = text["attention_mask"].to(device)
            pred = model({"text": text, "img": img})
            for i, _guid in enumerate(guid):
                if _guid not in Y_pred.keys():
                    Y_pred[_guid] = torch.tensor([0.0, 0.0, 0.0])
                Y_pred[_guid] += pred[i].to("cpu")
        for _guid in _data["guid"]:
            _data.loc[_data.guid == _guid, "tag"] = int(Y_pred[_guid].argmax())

    _data.loc[_data.tag == 0, "tag"] = "positive"
    _data.loc[_data.tag == 1, "tag"] = "neutral"
    _data.loc[_data.tag == 2, "tag"] = "negative"
    _data.to_csv("data/test.txt", index=False)
