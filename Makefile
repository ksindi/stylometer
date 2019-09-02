export MODEL_NAME:=uncased_L-12_H-768_A-12
export MODEL_PATH:=bert_models/${MODEL_NAME}
export DATA_PATH:=data

local-init: download-model

download-model:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} ./download_model.sh

merge-tweets:
	awk 'FNR==1 && NR!=1{next;}{print}' ./tweets/*.csv > ./data/data.csv

bert:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up --build -d bert-server

train:
	MODEL_NAME=${MODEL_NAME} MODEL_PATH=${MODEL_PATH} docker-compose up --build bert-server train

.PHONY: tweets
tweets:
	docker-compose up --build tweets

clean:
	docker-compose down --remove-orphans || true
	docker network prune --force
	docker system prune --volumes --force
