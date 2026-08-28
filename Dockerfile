FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard

RUN pip install --no-cache-dir .

RUN mkdir -p data/raw data/processed outputs

ENTRYPOINT ["python", "-m", "traffic_intelligence"]
CMD ["--help"]
