FROM python:3.6-slim-buster

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "01_download_dataset.py"]