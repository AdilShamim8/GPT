# Multi-stage production Dockerfile for nano-gpt-prod
FROM python:3.10-slim as builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.10-slim as runtime

WORKDIR /app

# Create non-root user for container security
RUN groupadd -r gptuser && useradd -r -g gptuser -m -d /home/gptuser gptuser

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .
RUN pip install --no-deps -e .

RUN chown -R gptuser:gptuser /app
USER gptuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["python", "serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
