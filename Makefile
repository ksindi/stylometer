export MODEL_NAME:=uncased_L-12_H-768_A-12
export MODEL_PATH:=bert_models/${MODEL_NAME}
export DATA_PATH:=data

local-init: download-model

download-dataset:
	DATA_PATH=${DATA_PATH} docker-compose up --build data

download-model:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} ./download_model.sh

build:
		docker-compose --build

bert-server:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up --build -d bert-server

train:
	docker-compose up --build train

local:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up --build bert-server

clean:
	docker-compose down --remove-orphans || true
	docker network prune --force
	docker system prune --volumes --force
