FROM python:3.10

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

WORKDIR /workspace

COPY pyproject.toml uv.lock config.yaml ./
COPY models ./models

RUN uv sync --frozen

ENV PATH="/venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "vllm", "serve", "--config", "config.yaml"]
