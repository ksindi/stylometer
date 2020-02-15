# Stylometer

Use BERT to perform stylometry using triplet loss.

## Requirements

- Docker
- docker-compose

## Setting up

Download BERT model and dataset:

```Makefile
make local-init
```

## Training

```Makefile
make train
```

## Evaluating

```Makefile
EVAL_DATASET=/some/file make eval
```

## Credit

Big thanks to the following projects for making this one easy to implement:
- https://github.com/hanxiao/bert-as-service
- https://github.com/omoindrot/tensorflow-triplet-loss
