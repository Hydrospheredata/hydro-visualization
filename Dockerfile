FROM python:3.8.9-slim-buster AS build

RUN apt-get update && \
    apt-get install -y -q build-essential git

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

RUN python -m venv /venv

WORKDIR /app/

COPY . .

RUN printf '{"name": "visualization", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json

RUN poetry export -f requirements.txt | /venv/bin/pip install -r /dev/stdin

RUN poetry build && /venv/bin/pip install dist/*.whl

FROM python:3.8.9-slim-buster

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV PATH=/home/app/.local/bin:$PATH

ENV APP_PORT=5000
EXPOSE ${APP_PORT}
ENV GRPC_PORT=5003
EXPOSE ${GRPC_PORT}
COPY --from=build --chown=app:app /root/.local /home/app/.local
COPY --chown=app:app app/ /app
COPY --from=build --chown=app:app buildinfo.json /app/buildinfo.json

WORKDIR /app