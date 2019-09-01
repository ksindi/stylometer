# Stylometer

Use BERT to perform stylometry using triplet loss.

## Requirements

- Docker
- docker-compose

## Setting up

#### Download BERT model and dataset:

```Makefile
make download-model
```

#### Get tweets:

Scrape tweets from twitter accounts. A CSV will be created in the form:

```csv
timestamp,raw_text,username
2015-12-21 21:33:41,very interesting thought,username1
```

You can modify the tweet usernames bu editing `twitter_handles.txt`.

```Makefile
make tweets
```

#### Create tfrecords:

Create tfrecords that model will call.
It will spin up a [BERT server](https://github.com/hanxiao/bert-as-service)
and encode the text into a 768 length numpy ndarray that is serialzied to a
tfrecord.

```Makefile
make tfrecords
```

## Training

```Makefile
make train
```

## Evaluating and Visualizing

```Makefile
EVAL_DATASET= make eval
```

## Testing

```Makefile
make test
```

## Credit

Big thanks to the following projects for making this one easy to implement:
- https://github.com/hanxiao/bert-as-service
- https://github.com/omoindrot/tensorflow-triplet-loss
