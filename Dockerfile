FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY config ./config
COPY data ./data
COPY src ./src
COPY tests ./tests
RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "lab-cli"]
CMD ["--help"]
