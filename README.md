<p align="center">
<h1 align="center">Multimodal Sentiment Analysis</h1>
<div align="center">
<img src="README.assets\model.drawio.svg", width="600">
</div>
</p>



### project structure

```shell
.
├── README.md
├── config
│   └── config.json
├── data
│   ├── data // images and texts
│   ├── test_without_label.txt
│   └── train.txt
├── main.py
├── models
│   ├── __init__.py
│   └── model.py
├── output // to save loss and accuracy record
└── utils
    ├── __init__.py
    ├── data_loader.py
    └── train.py
```

### start

```shell
python main.py --epochs [EPOCHS] --mode [MODE] --device [DEVICE] --verbose [VERBOSE]
```

- `epochs`: int, numbers of epochs for training, default 10
- `mode`:
  - `train_and_test` [default]
    - generates `models/mm_model.pt`, `data/test.txt`, `output/loss_and_acc.csv`
  - `train_only`
    - generates `models/mm_model.pt`
  - `test_only`
    - models/mm_model.pt needed
    - generates `data/test.txt`, `output/loss_and_acc.csv`
- `device`:
  - default: the device you have
- `verbose`:
  - show training progress in detail, default `True`
