FROM python:3.8.9-slim-buster as python-base
ENV POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"


FROM python-base AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y -q build-essential \
    git \
    curl

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
RUN mv /root/.poetry $POETRY_PATH
RUN python -m venv $VENV_PATH
RUN poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv

COPY version version
COPY . ./
RUN printf '{"name": "visualization", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json


FROM python-base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV APP_PORT=5000
EXPOSE ${APP_PORT}
ENV GRPC_PORT=5003
EXPOSE ${GRPC_PORT}

COPY --chown=app:app hydro_viz/ /hydro_viz
COPY --chown=app:app start.sh /hydro_viz/start.sh

WORKDIR /hydro_viz

COPY --from=build $VENV_PATH $VENV_PATH
COPY --from=build --chown=app:app buildinfo.json buildinfo.json
COPY . ./

ENTRYPOINT ["bash", "/hydro_viz/start.sh"] 
