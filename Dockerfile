# The Horizon Broker - runtime image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY horizon_broker ./horizon_broker
COPY config ./config

RUN pip install --no-cache-dir .

RUN mkdir -p /var/lib/horizon-broker && \
    useradd --create-home --uid 10001 horizon && \
    chown -R horizon:horizon /var/lib/horizon-broker
USER horizon

ENV JOURNAL_PATH=/var/lib/horizon-broker/journal.jsonl
EXPOSE 8791

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8791/healthz', timeout=2).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "horizon_broker"]
CMD ["serve", "--config", "/app/config/broker.yaml"]
