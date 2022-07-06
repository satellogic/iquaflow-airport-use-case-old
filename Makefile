ifndef DS_VOLUME
	DS_VOLUME=/Data/share
endif

ifndef NB_PORT
	NB_PORT=8888
endif

ifndef MLF_PORT
	MLF_PORT=5000
endif

help:
	@echo "build -- builds the docker image"

build:
	docker build -t airport .
	./download.sh

dockershell:
	docker run --rm --name air --gpus all -p 9197:9197 \
	-v $(shell pwd):/airport -v $(DS_VOLUME):/Data/share \
	-it airport

notebookshell:
	docker run --gpus all --privileged -itd --rm -p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/airport -v $(DS_VOLUME):/Data/share \
	airport \
	conda run -n airport-env jupyter notebook \
	--NotebookApp.token='AIRPORT' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
        --port=${NB_PORT}

mlflow:
	docker run --privileged -itd --rm -p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/airport -v $(DS_VOLUME):/Data/share \
	airport \
	conda run -n airport-env mlflow ui --host 0.0.0.0:${MLF_PORT}
