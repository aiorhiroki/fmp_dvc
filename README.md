# quick start

## Prerequisite

[install poetry](https://python-poetry.org/docs/#installation)

## Segmentation Example

### download dataset
```
git clone https://github.com/alexgkendall/SegNet-Tutorial ./seg_data
```

### Train example
```
poetry install
nohup poetry run python -u train_seg.py > train.out &
tail train.out -f
```
