FROM python:3.8-buster

RUN apt-get -y update && apt -y full-upgrade && apt-get -y install apt-utils

WORKDIR /workspace
COPY ./implementation/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY ./implementation /workspace

RUN chmod +x /workspace/run.sh

ENTRYPOINT ["python3", "/workspace/mlcube.py"]
