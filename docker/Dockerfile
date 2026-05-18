# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml ./
# CPU-only torch wheels to keep image small for the demo. GPU image is separate.
RUN pip install --upgrade pip \
    && poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app models ./models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "football_tracker.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
