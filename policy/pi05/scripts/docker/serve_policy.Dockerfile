# Build from the RoboSynChallenge root.
# docker build -f policy/pi05/scripts/docker/serve_policy.Dockerfile -t robosyn-pi05 .
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY robosynchallenge/VERSION robosynchallenge/VERSION
RUN uv sync --locked --python 3.12 --no-install-project --no-cache

COPY launch/openpi.sh launch/openpi.sh
COPY robosynchallenge/__init__.py robosynchallenge/hub.py robosynchallenge/
COPY policy/openpi_v30.py policy/openpi_v30.py
COPY policy/pi05/src policy/pi05/src
COPY policy/pi05/packages/openpi-client/src policy/pi05/packages/openpi-client/src
COPY policy/pi05/scripts/serve_policy.py policy/pi05/scripts/serve_policy.py

# The server uses the same locked dependency groups and vendored OpenPI code.
ENV UV_NO_SYNC=1 JAX_PLATFORMS=cuda
EXPOSE 8000
ENTRYPOINT ["bash", "launch/openpi.sh", "pi05", "serve"]
