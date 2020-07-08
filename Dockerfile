FROM python:3.8.2-slim-buster

RUN apt-get update && \
    apt-get install -y git build-essential

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

COPY --chown=app:app requirements.txt /requirements.txt
RUN pip3 install --user -r /requirements.txt

ENV PATH=/home/app/.local/bin:$PATH

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}

COPY --chown=app:app . /app

WORKDIR /app