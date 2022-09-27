# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /whycry

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]