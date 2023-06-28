import argparse

import torch
from torch import nn
from transformers.optimization import Adafactor, AdafactorSchedule

from models.model import MM_Model
from utils.data_loader import test_data, train_data
from utils.train import test, train

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def _train(
    model,
    lr=1e-3,
    weight_decay=0,
    train_loss=[],
    train_acc=[],
    val_loss=[],
    val_acc=[],
    verbose=True,
    epochs=10,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None,
        weight_decay=weight_decay,
    )
    lr_scheduler = AdafactorSchedule(optimizer)
    train_loader, val_loader = train_data()
    train(
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
        epochs=epochs,
        verbose=verbose,
    )


def _test(model):
    test_loader = test_data()
    test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train_and_test")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--verbose", type=str, default="True")
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    if device == "cuda" and args.device == "cpu":
        device = "cpu"

    model = MM_Model().to(device)

    if args.mode == "train_and_test":
        from transformers.optimization import Adafactor, AdafactorSchedule
        from utils.data_loader import test_data, train_data
        from utils.train import test, train

        _train(
            model,
            verbose=args.verbose == "True",
            epochs=args.epochs,
        )
        torch.save(model.state_dict(), "./models/mm_model.pt")
        _test(model)

    elif args.mode == "train_only":
        from transformers.optimization import Adafactor, AdafactorSchedule
        from utils.data_loader import train_data
        from utils.train import train

        _train(
            model,
            verbose=args.verbose == "True",
            epochs=args.epochs,
        )
        torch.save(model.state_dict(), "./models/mm_model.pt")
    elif args.mode == "test_only":
        from utils.data_loader import test_data
        from utils.train import test

        model.load_state_dict(torch.load("./models/mm_model.pt"))
        _test(model)
    else:
        print("Error")
