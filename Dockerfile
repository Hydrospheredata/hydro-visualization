FROM python:3.8.2-slim-buster AS build

RUN apt-get update && \
    apt-get install -y git build-essential

COPY requirements.txt requirements.txt
RUN pip3 install --user -r requirements.txt

FROM python:3.8.2-slim-buster

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV PATH=/home/app/.local/bin:$PATH

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}

COPY --from=build --chown=app:app /root/.local /home/app/.local
COPY --chown=app:app . /app

WORKDIR /app