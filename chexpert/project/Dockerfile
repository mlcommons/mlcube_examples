FROM ubuntu:18.04
MAINTAINER MLPerf MLBox Working Group

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	software-properties-common \
	python3-dev \
	curl \
	wget \
	libsm6 libxext6 libxrender-dev && \
	rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update

RUN apt-get install python3.7 -y

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3.7 get-pip.py && \
	rm get-pip.py

COPY ./requirements.txt project/requirements.txt 

RUN python3.7 -m pip install --upgrade pip

RUN python3.7 -m pip install --no-cache-dir -r project/requirements.txt

COPY . /project

WORKDIR /project

ENTRYPOINT ["python3.7", "mlcube.py"]