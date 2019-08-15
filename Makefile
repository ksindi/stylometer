export MODEL_NAME:=uncased_L-12_H-768_A-12
export MODEL_PATH:=bert_models/${MODEL_NAME}
export DATA_PATH:=data

local-init: download-model

download-dataset:
	DATA_PATH=${DATA_PATH} docker-compose up --build data

download-model:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} ./download_model.sh

train: clean
	docker-compose --build
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up -d bert-server
	sleep 10
	docker-compose up train

local:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up --build bert-server

clean:
	docker-compose down --remove-orphans || true
	docker network prune --force
	docker system prune --volumes --force
