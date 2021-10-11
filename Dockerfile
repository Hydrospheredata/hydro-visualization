# syntax=docker/dockerfile:1
FROM python:3.8.12-slim-bullseye as python-base
LABEL maintainer="support@hydrosphere.io"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6

ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"
RUN apt-get update && apt-get install -y -q --no-install-recommends \
    libgssapi-krb5-2>=1.18.3-6+deb11u1 \
    libk5crypto3>=1.18.3-6+deb11u1 \
    libkrb5-3>=1.18.3-6+deb11u1 \
    libkrb5support0>=1.18.3-6+deb11u1 \
    libssl1.1>=1.1.1 \
    openssl>=1.1.1 && \
    rm -rf /var/lib/apt/lists/*


FROM python-base AS build

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    UCF_FORCE_CONFOLD=1 

RUN apt-get update && \
    apt-get install -y -q --no-install-recommends \
    build-essential \
    curl \
    git && \
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && \
    mv /root/.poetry $POETRY_PATH && \
    python -m venv $VENV_PATH && \
    poetry config virtualenvs.create false && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv

ARG GIT_HEAD_COMMIT
ARG GIT_CURRENT_BRANCH
COPY . ./
RUN if [ -z "$GIT_HEAD_COMMIT" ] ; then \
    printf '{"name": "visualization", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json ; else \
    printf '{"name": "visualization", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$GIT_HEAD_COMMIT" "$GIT_CURRENT_BRANCH" "$(python --version)" >> buildinfo.json ; \
    fi


FROM python-base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

ENV APP_PORT=5000 \
    GRPC_PORT=5003

EXPOSE ${APP_PORT} ${GRPC_PORT}

WORKDIR /app

COPY --from=build $VENV_PATH $VENV_PATH
COPY --chown=app:app hydro_viz ./hydro_viz
COPY --chown=app:app start.sh start.sh
COPY --from=build --chown=app:app buildinfo.json buildinfo.json

ENTRYPOINT ["bash", "start.sh"] 
