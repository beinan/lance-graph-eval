FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       git \
       protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY datasets/README.md /app/datasets/README.md

RUN pip install --upgrade pip \
    && pip install -e .[all]

ENTRYPOINT ["lgeval"]
