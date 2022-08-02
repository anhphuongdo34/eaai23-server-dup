FROM python:3.9.12-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip3 install -r requirements.txt


COPY . .

CMD ["python3", "main.py"]