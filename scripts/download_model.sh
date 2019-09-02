#!/bin/sh
set -eu pipefail

echo "--- :house_with_garden: Downloading model $MODEL_NAME"

curl -Lf -o ${MODEL_PATH}.zip https://storage.googleapis.com/bert_models/2018_10_18/${MODEL_NAME}.zip \
  && unzip ${MODEL_PATH}.zip -d $(dirname ${MODEL_PATH})
