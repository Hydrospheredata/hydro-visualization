FROM python:3.7.4-slim-stretch

RUN apt-get update && \
    apt-get upgrade -y &&\
    apt-get install -y git

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip3 install -r requirements.txt

WORKDIR /app
EXPOSE 5000

COPY . .
