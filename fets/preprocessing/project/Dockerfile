# for a CPU app use this Dockerfile.
FROM python:3.8-buster

# fill in your info here
LABEL author="chuck@norris.org"
LABEL application="your application name"
LABEL maintainer="chuck@norris.org"
LABEL version="0.0.1"
LABEL status="beta"

# basic
RUN apt-get -y update && apt -y full-upgrade && apt-get -y install apt-utils wget git tar build-essential curl nano

# install all python requirements
WORKDIR /workspace
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copy all files
COPY ./ /workspace

RUN chmod +x /workspace/run.sh

ENTRYPOINT [ "python3", "/workspace/mlcube.py"]
