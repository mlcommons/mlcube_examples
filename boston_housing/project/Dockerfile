FROM python:3.6-slim-buster

#RUN sudo apt update && sudo apt search vim && sudo apt install vim

WORKDIR /workspace/boston_housing

COPY requirements.txt requirements.txt

RUN pip install --no-cache --disable-pip-version-check --default-timeout=1000 -U -r requirements.txt

COPY . .

RUN chmod +x ./run_and_time.sh

ENTRYPOINT ["python", "/workspace/boston_housing/mlcube.py"]