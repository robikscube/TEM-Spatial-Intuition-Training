FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv

FROM python:3.13.15-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26

ARG TEM_ACCELERATOR=cpu

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/tem-env \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/tem-env/bin:$PATH"

WORKDIR /app

COPY docker/pyproject.toml docker/uv.lock /app/docker/

RUN uv sync --project /app/docker --locked --no-dev --no-install-project --extra "$TEM_ACCELERATOR"

COPY scripts/*.py /app/scripts/
COPY docker/check_runtime.py docker/upstream.lock.json /app/docker/

RUN mkdir -p /runs /upstream /app/experiments && chown -R 10001:10001 /runs

USER 10001:10001

ENTRYPOINT ["python", "-B", "/app/docker/check_runtime.py"]
